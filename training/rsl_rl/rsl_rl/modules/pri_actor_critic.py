"""Privileged Actor-Critic architecture for DreamWaQ++-style navigation.

The actor encodes the navigation observation through three modalities:

* lidar rays as a point cloud with a PointNet encoder;
* proprioceptive history with an MLP-Mixer encoder;
* the local goal with a small MLP.

A confidence gate weights the modality embeddings before the fused actor
backbone and navigation heads. The policy mean is passed through the
differentiable CBF safety layer. When ``num_privileged_obs`` is configured,
the critic consumes that privileged vector; otherwise it falls back to the
actor representation so the class also works with the current non-privileged
SEA-Nav environment.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from .cbf_lse_layer import ExactLSECBFLayer


class PointNetEncoder(nn.Module):
    """Encode 2-D lidar rays as an unordered set of (x, y, range) points."""

    def __init__(self, output_dim=128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, output_dim), nn.ELU(),
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ELU(),
        )

    def forward(self, ranges, ray_angles):
        # ranges: [batch, rays], ray_angles: [rays]
        # 将每条激光距离转换为平面点，再对点特征做池化。
        x = ranges * torch.cos(ray_angles).unsqueeze(0)
        y = ranges * torch.sin(ray_angles).unsqueeze(0)
        points = torch.stack((x, y, ranges), dim=-1)
        point_features = self.point_mlp(points)
        pooled = torch.cat((point_features.amax(dim=1), point_features.mean(dim=1)), dim=-1)
        return self.pool_mlp(pooled)


class MLPMixerBlock(nn.Module):
    """Mix history tokens and proprioceptive channels with residual MLPs."""

    def __init__(self, history_len, channels, token_hidden_dim=None, channel_hidden_dim=None):
        super().__init__()
        token_hidden_dim = token_hidden_dim or history_len * 2
        channel_hidden_dim = channel_hidden_dim or channels * 2
        self.token_norm = nn.LayerNorm(channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(history_len, token_hidden_dim), nn.GELU(),
            nn.Linear(token_hidden_dim, history_len),
        )
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channel_hidden_dim), nn.GELU(),
            nn.Linear(channel_hidden_dim, channels),
        )

    def forward(self, x):
        # x: [batch, history, channels]
        # 先混合时间步，再混合每个时间步的特征通道。
        token_input = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mlp(token_input).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class MLPProprioEncoder(nn.Module):
    """Encode a fixed-length sequence of proprioceptive observations."""

    def __init__(self, proprio_dim, history_len, embed_dim=128, num_mixer_blocks=2):
        super().__init__()
        self.input_proj = nn.Linear(proprio_dim, embed_dim)
        self.mixer = nn.Sequential(*[
            MLPMixerBlock(history_len, embed_dim) for _ in range(num_mixer_blocks)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, proprio_history):
        # 将本体历史投影到统一维度，并取混合后的最新时刻特征。
        x = self.input_proj(proprio_history)
        x = self.mixer(x)
        return self.output_norm(x[:, -1])


class ConfidenceFusion(nn.Module):
    """Fuse modality embeddings using learned per-sample confidence weights."""

    def __init__(self, modality_dim=128, fused_dim=256, num_modalities=3):
        super().__init__()
        self.confidence = nn.Sequential(
            nn.Linear(modality_dim * num_modalities, 128), nn.ELU(),
            nn.Linear(128, num_modalities),
        )
        self.fusion = nn.Sequential(
            nn.Linear(modality_dim * (num_modalities + 1), fused_dim), nn.ELU(),
            nn.Linear(fused_dim, fused_dim), nn.ELU(),
        )

    def forward(self, modality_features):
        # Each modality has shape [batch, modality_dim].
        stacked = torch.stack(modality_features, dim=1)
        # 根据当前输入计算各模态权重，再融合加权特征。
        confidence = torch.softmax(self.confidence(torch.cat(modality_features, dim=-1)), dim=-1)
        weighted = torch.sum(stacked * confidence.unsqueeze(-1), dim=1)
        fused = self.fusion(torch.cat((*modality_features, weighted), dim=-1))
        return fused, confidence


class PriDifferentiableSafeActorCritic(nn.Module):
    """PointNet + MLP-Mixer actor with confidence fusion and optional privileged critic."""

    is_recurrent = False
    def __init__(self,  
                    num_actions,
                    actor_hidden_dims=[256, 256, 256],
                    critic_hidden_dims=[256, 256, 256],
                    encoder_hidden_dims=[512, 256, 128],
                    activation='elu',
                    init_noise_std=1.5,
                    num_props=12,
                    num_rays=41,
                    his_len=10,
                    num_privileged_obs=0,
                    ray_angle_min=-2.0 * torch.pi / 3.0,
                    ray_angle_max=2.0 * torch.pi / 3.0,
                 **kwargs):
        super().__init__()

        activation = get_activation(activation)
        
        self.his_len = his_len
        self.num_rays = num_rays
        self.num_props = num_props
        self.num_obs_one_step = num_rays + num_props + 2  # 2 for target x,y
        self.num_obs_hist = self.num_obs_one_step * his_len
        self.num_actions = num_actions
        self.num_privileged_obs = int(num_privileged_obs or 0)
        self.enable_shield = True

        # 激光射线角度用于把距离还原为二维点云。
        ray_angles = torch.linspace(ray_angle_min, ray_angle_max, num_rays)
        self.register_buffer("ray_angles", ray_angles, persistent=True)
        self.pointnet = PointNetEncoder(output_dim=128)
        scan_encoder_dims = [his_len * 128] + list(encoder_hidden_dims)
        if scan_encoder_dims[-1] != 128:
            scan_encoder_dims.append(128)
        scan_encoder_layers = []
        for in_dim, out_dim in zip(scan_encoder_dims[:-1], scan_encoder_dims[1:]):
            scan_encoder_layers.extend((nn.Linear(in_dim, out_dim), activation))
        self.scan_temporal_encoder = nn.Sequential(*scan_encoder_layers)
        self.proprio_encoder = MLPProprioEncoder(num_props, his_len, embed_dim=128)
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 64), activation,
            nn.Linear(64, 128), activation,
        )
        self.fusion = ConfidenceFusion(modality_dim=128, fused_dim=256)

        # 1. Shared Backbone
        backbone_layers = []
        backbone_layers.append(nn.Linear(256, actor_hidden_dims[0]))
        backbone_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                pass
            else:
                backbone_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                backbone_layers.append(activation)
        self.backbone = nn.Sequential(*backbone_layers)

        # 2. Navigation Head
        self.nav_head = nn.Sequential(
            nn.Linear(actor_hidden_dims[-1], 128),
            activation,
            nn.Linear(128, num_actions)
        )

        # Critic: privileged input when configured, actor feature otherwise.
        critic_input_dim = self.num_privileged_obs or actor_hidden_dims[-1]
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # 3. Adaptive Safety Head (Alpha Head)
        self.alpha_head = nn.Sequential(
            nn.Linear(actor_hidden_dims[-1], 64),
            activation,
            nn.Linear(64, 1)
        )

        # 4. Closed-form CBF Layer
        self.cbf_layer = ExactLSECBFLayer(num_rays=num_rays)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    

    def extract(self, observations):
        # 将展平观测还原成历史帧，并拆出本体、激光和目标信息。
        obs_hist = observations.reshape(-1, self.his_len, self.num_obs_one_step)
        props_hist = obs_hist[:, :, :self.num_props]
        rays_hist = obs_hist[:, :, self.num_props:self.num_props + self.num_rays]
        goals = obs_hist[:, -1, -2:]
        rays_real_hist = torch.exp2(rays_hist).clamp(min=0.1, max=5.0)
        return props_hist, rays_hist, rays_real_hist, goals

    def encode_actor_observations(self, observations):
        props_hist, rays_hist, rays_real_hist, goals = self.extract(observations)
        batch_size = observations.shape[0]

        # PointNet编码每帧点云，时序MLP汇总扫描历史。
        scan_embeds = self.pointnet(
            rays_real_hist.reshape(batch_size * self.his_len, self.num_rays),
            self.ray_angles,
        ).reshape(batch_size, self.his_len, 128)
        scan_embed = self.scan_temporal_encoder(scan_embeds.reshape(batch_size, -1))
        proprio_embed = self.proprio_encoder(props_hist)
        goal_embed = self.goal_encoder(goals)
        # 融合点云、本体和目标特征，返回Actor主干特征及最新射线。
        fused, self.modality_confidence = self.fusion((scan_embed, proprio_embed, goal_embed))
        return self.backbone(fused), rays_hist[:, -1]

    def forward(self, observations):
        """
        Directly return the safety mean used to build the distribution.
        Following the original paper concept, CBF is the final layer of the network.
        """
        shared_features, rays = self.encode_actor_observations(observations)
        
        # 2. Dual-head parallel inference
        # 导航头生成原始动作，安全头生成CBF参数。
        u_bar = self.nav_head(shared_features)
        alpha_raw = self.alpha_head(shared_features)
        
        # 3. Calculate adaptive parameter \alpha, using softplus to ensure \alpha > 0 mathematically
        rays_real = torch.exp2(rays).clamp(min=0.1, max=5.0)
        alpha = F.softplus(alpha_raw)
        self.alpha = alpha 
        
        # 4. Get u_s through differentiable safety layer
        u_s = self.cbf_layer(u_bar, rays_real, alpha)
        
        # Save u_bar and u_s for calculating Intervention Loss
        self.u_bar = u_bar
        self.u_s = u_s
        
        return u_s
    
    def update_distribution(self, observations):
        mean = self.forward(observations)
        self.distribution = Normal(mean, mean*0. + self.std)
        self.mean = mean

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.sample()
        return actions

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations, privileged_observations=None, **kwargs):
        if self.num_privileged_obs:
            # 配置了特权维度时，Critic只使用特权观测估值。
            critic_obs = privileged_observations if privileged_observations is not None else observations
            if critic_obs.shape[-1] != self.num_privileged_obs:
                raise ValueError(
                    f"Critic expects {self.num_privileged_obs} privileged features, "
                    f"received {critic_obs.shape[-1]}."
                )
            critic_input = critic_obs
        else:
            # 没有特权观测时，使用Actor编码特征作为Critic输入。
            critic_input, _ = self.encode_actor_observations(observations)
        value = self.critic(critic_input)
        return value
    
    def act_inference(self, observations, **kwargs):
        """ Used effectively during evaluation or deployment, without exploration noise """
        u_s = self.forward(observations)
        return u_s


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
