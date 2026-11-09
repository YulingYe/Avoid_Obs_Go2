import cv2
import numpy as np
from nav_msgs.msg import Odometry

from ros_base.configs.camera import get_realsense_d435i
from ros_base.nodes.base_node import BaseNode
from ros_base.nodes.camera.pose_processor import PoseProcessor


class CamPoseSubNode(BaseNode):
    def __init__(
        self,
        verbose: bool = True,
        log_freq: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.verbose = verbose
        self.log_freq = log_freq

        self.odom_sub = self.create_subscription(
            Odometry,
            "/visual_slam/tracking/odometry",
            self._pose_callback,
            1,
        )
        # Initialize PoseProcessor with your specific hardware params
        self.pose_processor = PoseProcessor(cfg=get_realsense_d435i(align_depth=True, compressed=False))

        self.counter = 0

    def _pose_callback(self, msg: Odometry):
        self.pose_processor.update_pose_from_odom(msg)
        # Log Robot Base pose in World Frame every 50 messages
        self.get_robot_base_world_pose()
        self.counter += 1

    def get_robot_base_world_pose(self):
        base_pos, base_rpy = self.pose_processor.get_robot_base_world_pose()
        if self.verbose and self.counter % self.log_freq == 0:
            self.logger.info(
                f"Base Position (World Frame): x={base_pos[0]:.2f}, y={base_pos[1]:.2f}, z={base_pos[2]:.2f} | Base"
                f" Orientation (rpy in radians): roll={base_rpy[0]:.2f}, pitch={base_rpy[1]:.2f}, yaw={base_rpy[2]:.2f}"
            )
        return base_pos, base_rpy


if __name__ == "__main__":
    import rclpy

    from ros_base.utils.args_debug import add_debug_mode

    parser = add_debug_mode(listen_port=9999)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--log_freq", type=int, default=50, help="Logging frequency for pose updates.")

    args = parser.parse_args()

    rclpy.init()
    cam_pose_node = CamPoseSubNode(node_name="cam_pose_sub_node", verbose=args.verbose, log_freq=args.log_freq)
    try:
        cam_pose_node.start_spin_standalone()
    except KeyboardInterrupt:
        pass
    finally:
        cam_pose_node.release_resources()
        if rclpy.ok():
            cam_pose_node.destroy_node()
            rclpy.shutdown()
