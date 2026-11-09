"""Export a safe navigation actor-critic checkpoint without Isaac Gym.

Run with the sea_nav Python environment. Activation and non-buffer CBF
parameters are not stored in the checkpoint: defaults below match the current
training source. Override them if the training run used other values.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "rsl_rl"))
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.modules.pri_actor_critic import PriDifferentiableSafeActorCritic


class ExportPolicy(nn.Module):
    """Full deterministic policy; Pow replaces Exp2 for ONNX compatibility."""

    def __init__(self, policy):
        super().__init__()
        self.encoder = policy.encoder
        self.backbone = policy.backbone
        self.nav_head = policy.nav_head
        self.alpha_head = policy.alpha_head
        self.cbf_layer = policy.cbf_layer
        self.step_size = policy.num_obs_one_step
        self.num_props = policy.num_props
        self.num_rays = policy.num_rays

    def forward(self, observations):
        current = observations[:, -self.step_size:]
        latent = self.encoder(observations)
        features = self.backbone(torch.cat((current, latent), dim=-1))
        nominal = self.nav_head(features)
        alpha = torch.nn.functional.softplus(self.alpha_head(features))
        rays_log2 = current[:, self.num_props:self.num_props + self.num_rays]
        return self.cbf_layer(nominal, torch.pow(2.0, rays_log2), alpha)


class PriExportPolicy(nn.Module):
    """OARMM PointNet + MLP-Mixer policy with deterministic CBF output."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations):
        policy = self.policy
        history = observations.reshape(-1, policy.his_len, policy.num_obs_one_step)
        props = history[:, :, :policy.num_props]
        rays_log2 = history[:, :, policy.num_props:policy.num_props + policy.num_rays]
        batch_size = observations.shape[0]
        rays = torch.pow(2.0, rays_log2).clamp(min=0.1, max=5.0)
        goals = history[:, -1, -2:]

        scan_features = policy.pointnet(
            rays.reshape(batch_size * policy.his_len, policy.num_rays),
            policy.ray_angles,
        ).reshape(batch_size, policy.his_len, 128)
        scan_features = policy.scan_temporal_encoder(scan_features.reshape(batch_size, -1))
        proprio_features = policy.proprio_encoder(props)
        goal_features = policy.goal_encoder(goals)
        fused, _ = policy.fusion((scan_features, proprio_features, goal_features))
        features = policy.backbone(fused)

        nominal = policy.nav_head(features)
        alpha = torch.nn.functional.softplus(policy.alpha_head(features))
        return policy.cbf_layer(nominal, rays[:, -1], alpha)


def hidden_dims(state, prefix, exclude_last=False):
    weights = sorted(
        ((int(k.split(".")[1]), v) for k, v in state.items()
         if k.startswith(prefix + ".") and k.endswith(".weight")),
        key=lambda pair: pair[0],
    )
    dims = [int(v.shape[0]) for _, v in weights]
    return dims[:-1] if exclude_last else dims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--activation", default="elu",
                        choices=["elu", "relu", "selu", "crelu", "lrelu", "tanh", "sigmoid"])
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--safe-distance", type=float, default=0.20)
    parser.add_argument("--damping-factor", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              checkpoint.parent / "exported" / (checkpoint.stem + ".onnx"))
    report_path = output.with_suffix(".json")
    if not args.force and (output.exists() or report_path.exists()):
        parser.error("Output already exists; choose another output or use --force")
    torch.set_num_threads(1)
    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = loaded["model_state_dict"]
    if "backbone.0.weight" not in state or "cbf_layer.ray_unit_vectors" not in state:
        parser.error("Checkpoint does not contain a supported CBF actor-critic state")

    num_actions = int(state["std"].numel())
    num_rays = int(state["cbf_layer.ray_unit_vectors"].shape[0])
    if "pointnet.point_mlp.0.weight" in state:
        # OARMM policy: PointNet scans + MLP-Mixer proprioceptive history.
        model_kind = "PriDifferentiableSafeActorCritic"
        num_props = int(state["proprio_encoder.input_proj.weight"].shape[1])
        history_len = int(state["proprio_encoder.mixer.0.token_mlp.0.weight"].shape[1])
        step_size = num_props + num_rays + 2
        input_size = history_len * step_size
        actor_dims = hidden_dims(state, "backbone")
        critic_input = int(state["critic.0.weight"].shape[1])
        privileged_dim = 0 if critic_input == actor_dims[-1] else critic_input
        policy = PriDifferentiableSafeActorCritic(
            num_actions=num_actions,
            actor_hidden_dims=actor_dims,
            critic_hidden_dims=hidden_dims(state, "critic", exclude_last=True),
            encoder_hidden_dims=hidden_dims(state, "scan_temporal_encoder", exclude_last=True),
            activation=args.activation, num_props=num_props, num_rays=num_rays,
            his_len=history_len, num_privileged_obs=privileged_dim,
        ).cpu().eval()
        wrapper = PriExportPolicy(policy).eval()
    elif "encoder.0.weight" in state:
        # Legacy CBF actor-critic: vector history encoder + current-step actor.
        model_kind = "DifferentiableSafeActorCritic"
        input_size = int(state["encoder.0.weight"].shape[1])
        num_props = 12
        history_len = 10
        step_size = int(state["backbone.0.weight"].shape[1] - state["encoder.6.weight"].shape[0])
        policy = DifferentiableSafeActorCritic(
            num_actions=num_actions,
            actor_hidden_dims=hidden_dims(state, "backbone"),
            critic_hidden_dims=hidden_dims(state, "critic", exclude_last=True),
            encoder_hidden_dims=hidden_dims(state, "encoder", exclude_last=True),
            activation=args.activation, num_props=num_props, num_rays=num_rays,
            his_len=history_len,
        ).cpu().eval()
        wrapper = ExportPolicy(policy).eval()
    else:
        parser.error("Unrecognized actor-critic checkpoint architecture")

    if (input_size != 550 or step_size != 55 or num_rays != 41
            or num_actions != 3 or num_props != 12 or history_len != 10):
        parser.error("Checkpoint does not match the supported SEA-Nav 550 -> 3 interface")

    policy.eval()
    policy.load_state_dict(state, strict=True)
    policy.cbf_layer.kappa = args.kappa
    policy.cbf_layer.d_safe = args.safe_distance
    policy.cbf_layer.damping_factor = args.damping_factor
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, torch.zeros(1, input_size), str(output), opset_version=17,
        input_names=["observations"], output_names=["velocity_command"],
        dynamic_axes={"observations": {0: "batch"}, "velocity_command": {0: "batch"}},
    )
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(2000)
    max_error = 0.0
    total = 0
    for batch in (1, 4, 16):
        for fixed_distance in (None, 0.1, 0.2, 3.0):
            obs = rng.uniform(-1, 1, (batch, history_len, step_size)).astype(np.float32)
            distances = rng.uniform(0.1, 3.0, (batch, history_len, num_rays))
            if fixed_distance is not None:
                distances.fill(fixed_distance)
            obs[:, :, num_props:num_props + num_rays] = np.log2(distances)
            obs[:, :, -2:] *= 5.0
            flat = obs.reshape(batch, input_size)
            with torch.no_grad():
                expected = policy.act_inference(torch.from_numpy(flat)).numpy()
                wrapped = wrapper(torch.from_numpy(flat)).numpy()
            actual = session.run(None, {"observations": flat})[0]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(wrapped, expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            total += batch
    vectors = state["cbf_layer.ray_unit_vectors"].numpy()
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "iteration": loaded.get("iter"),
        "policy_class": model_kind,
        "input_shape": ["batch", input_size], "output_shape": ["batch", 3],
        "activation_assumed": args.activation,
        "cbf_nonbuffer_parameters": {"kappa": args.kappa, "d_safe": args.safe_distance,
                                     "damping_factor": args.damping_factor},
        "checkpoint_cbf_fov_degrees": float(
            math.degrees(math.atan2(vectors[-1, 1], vectors[-1, 0]) -
                         math.atan2(vectors[0, 1], vectors[0, 0]))),
        "validation_samples": total, "max_absolute_error": max_error,
        "note": "Export equivalence does not verify training/deployment observation semantics or navigation quality.",
    }
    report_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("Exported:", output)
    print("Validation: %d samples, maximum absolute error %.8g" % (total, max_error))
    print("Report:", report_path)


if __name__ == "__main__":
    main()
