import cv2
import numpy as np
from geometry_msgs.msg import Point32, PolygonStamped
from kalman.simulation.base_scene import CameraParameters
from kalman.tracker.cartesian.visual_odometry_batch_kf import (
    CartesianVisualOdometryBatchTrackerKF,
)
from nav_msgs.msg import Odometry

from ros_base.configs.camera import get_realsense_d435i
from ros_base.nodes.base_node import BaseNode
from ros_base.utils.args_debug import add_debug_mode
from ros_base.utils.camera_trans import CameraTrans
from ros_base.utils.decorators import profile_latency
from ros_base.utils.math_utils import *

INVALID_MEAS = -1.0


class KFSigmaNode(BaseNode):
    def __init__(
        self,
        tracker: object = CartesianVisualOdometryBatchTrackerKF,
        cam_cfg=None,
        history_size: int = 100,
        sim_run: bool = False,
        frame_id: str = "base",
        num_points: int = 7,
        input_type: str = "xyz",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_points = num_points
        self.input_type = input_type

        camera_params = CameraParameters(
            focal_length=(cam_cfg.intrinsics.fx, cam_cfg.intrinsics.fy),
            principal_point=(cam_cfg.intrinsics.cx, cam_cfg.intrinsics.cy),
            image_size=(cam_cfg.intrinsics.width, cam_cfg.intrinsics.height),
            distortion_coeffs=getattr(cam_cfg.intrinsics, "D", None),
        )

        # Initialize tracker for N points (Sigma Points)
        self.tracker: CartesianVisualOdometryBatchTrackerKF = self.create_tracker(tracker, camera_params, history_size)
        self.sim_run = sim_run
        self.frame_id = frame_id
        self.last_valid_time = 0.0
        self.camera_trans = CameraTrans(cfg=cam_cfg)

        # Subscribe to PolygonStamped for Sigma Points (N points)
        self.sigma_sub = self.create_subscription(
            PolygonStamped, "/geometry_msgs/sigma_points", self._perception_callback, 1
        )

        self.sigma_filter_pub = self.create_publisher(PolygonStamped, "/geometry_msgs/sigma_points_filtered", 1)
        self.sigma_filter_msg = PolygonStamped()
        self.create_timer(0.02, self.publish_sigma_filter)  # 50 Hz

        self.odom_sub = self.create_subscription(
            Odometry,
            "/visual_slam/tracking/odometry",
            self._pose_callback,
            1,
        )

        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_quat_wxyz = np.quaternion(1, 0, 0, 0)

        if self.sim_run:
            self.pose_sub = self.create_subscription(Odometry, "/mujoco/odom", self._pose_callback, 10)
            self.frame_id = "base"

        self.T_bc = np.eye(4)
        self.T_bc[:3, :3] = self.camera_trans.R.T
        self.T_bc[:3, 3] = self.camera_trans.T

        R_ros_to_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        self.T_ros_to_opt = np.eye(4)
        self.T_ros_to_opt[:3, :3] = R_ros_to_opt

        self.det_lag_points = []  # List of [u, v, z]
        self.is_target_valid = False
        self.last_T_wc = None

    def create_tracker(self, tracker_class, camera_params, history_size):
        tracker = tracker_class(
            camera_params, num_points=self.num_points, history_size=history_size, dt=0.02, input_type=self.input_type
        )
        return tracker

    def _get_valid_data(self, points):
        """Check measurement validity"""
        if len(points) != self.num_points:
            self.det_lag_points = []
            return False

        valid_points = []
        for p in points:
            # p has x, y, z in camera frame
            if p.z == INVALID_MEAS:  # Check Z instead of X for XYZ mode
                self.det_lag_points = []
                return False
            valid_points.append([p.x, p.y, p.z])

        self.det_lag_points = valid_points
        return True

    def _handle_validity(self, timestamp):
        if not self.is_target_valid:
            pass
            # self.tracker.reset()
        else:
            self.last_valid_time = timestamp
            if not self.tracker.initialized:
                self.tracker.initialize(
                    points=self.det_lag_points,
                    timestamp=timestamp,
                )
                self.logger.info("KF Sigma Tracker initialized.")

    @profile_latency(
        cycle_threshold_ms=250.0,
        process_threshold_ms=20.0,
        log_interval_s=10.0,
        debug=True,
        check_capture_latency=False,
    )
    def _pose_callback(self, msg: Odometry):
        self.last_odom_stamp = msg.header.stamp
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self.tracker.initialized:
            return

        if hasattr(self.tracker.kf, "last_timestamp") and timestamp <= self.tracker.kf.last_timestamp:
            return

        tx = msg.pose.pose.position.x
        ty = msg.pose.pose.position.y
        tz = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        xw = qx * qw
        yw = qy * qw
        zw = qz * qw

        pos_odom = np.array([tx, ty, tz])

        R_odom = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
                [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
                [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)],
            ]
        )

        T_msg = np.eye(4)
        T_msg[:3, :3] = R_odom
        T_msg[:3, 3] = pos_odom

        if self.frame_id == "camera":
            T_wc = T_msg @ self.T_ros_to_opt
        else:
            T_wc = T_msg @ self.T_bc

        self.last_T_wc = T_wc
        self.tracker.predict(current_pose_wc=T_wc, timestamp=timestamp)

    @profile_latency(
        cycle_threshold_ms=250.0,
        process_threshold_ms=100.0,
        log_interval_s=10.0,
        debug=True,
        check_capture_latency=False,
    )
    def _perception_callback(self, msg: PolygonStamped):
        self.last_msg_stamp = msg.header.stamp
        capture_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.is_target_valid = self._get_valid_data(msg.polygon.points)
        self._handle_validity(capture_time)

        if self.is_target_valid:
            try:
                self.tracker.update(timestamp=capture_time, points=self.det_lag_points, debug=False)
            except ValueError as e:
                self.logger.warn(f"Tracker update failed: {e}")
        # else:
        #     # When invalid, only perform prediction if initialized
        #     if self.tracker.initialized and self.last_T_wc is not None:
        #         self.tracker.predict(current_pose_wc=self.last_T_wc, timestamp=capture_time)

    def publish_sigma_filter(self):
        est_states = self._get_est_states()

        if not self.is_target_valid:
            # If target invalid, collapse all points to the main point (index 0)
            main_point = est_states[0]
            est_states = [main_point] * self.num_points

        if hasattr(self, "last_odom_stamp"):
            self.sigma_filter_msg.header.stamp = self.last_odom_stamp
        else:
            self.sigma_filter_msg.header.stamp = self.get_clock().now().to_msg()

        self.sigma_filter_msg.header.frame_id = "camera_frame"

        points_msg = []
        for state in est_states:
            p = Point32()
            if state is None:
                p.x = float(INVALID_MEAS)
                p.y = float(INVALID_MEAS)
                p.z = float(INVALID_MEAS)
            else:
                # Publish Camera Frame Coordinates (X, Y, Z)
                p.x = float(state[0])
                p.y = float(state[1])
                p.z = float(state[2])
            points_msg.append(p)

        self.sigma_filter_msg.polygon.points = points_msg
        self.sigma_filter_pub.publish(self.sigma_filter_msg)

        # Call visualization
        # visualize_p_img(
        #     est_pos=est_states,
        #     gd_pos=self.det_lag_points if self.is_target_valid else None,
        #     camera=self.tracker.camera,
        #     width=self.tracker.camera.width,
        #     height=self.tracker.camera.height,
        # )

    def _get_est_states(self):
        if not self.tracker.initialized:
            return [None] * self.num_points
        return self.tracker.get_state()


def visualize_p_img(est_pos, gd_pos, camera, width, height):
    """
    est_pos: list of [X, Y, Z] (filtered output)
    gd_pos: list of [X, Y, Z] (raw measurements)
    """
    # Create a dark canvas for visualization
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Helper for projection (using the tracker's pinhole camera logic)
    def project(pt):
        if pt is None or pt[0] == INVALID_MEAS:
            return None
        X, Y, Z = pt
        if Z <= 0.05:  # Too close or negative depth
            return None
        u = (X * camera.fx / Z) + camera.cu
        v = (Y * camera.fy / Z) + camera.cv
        return (int(u), int(v))

    # 1. Draw Ground Truth / Raw Measurements (Green solid circles)
    if gd_pos is not None:
        for pt in gd_pos:
            uv = project(pt)
            if uv and (0 <= uv[0] < width and 0 <= uv[1] < height):
                cv2.circle(vis_img, uv, 4, (0, 255, 0), -1)

    # 2. Draw Estimated / Filtered Points (White Ring circles)
    if est_pos is not None:
        for pt in est_pos:
            uv = project(pt)
            if uv and (0 <= uv[0] < width and 0 <= uv[1] < height):
                # Thin white ring around the estimation
                cv2.circle(vis_img, uv, 6, (255, 255, 255), 1)

    # 3. Add Legend/Overlay Text
    cv2.putText(vis_img, "Green: Raw Input (Measurement)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_img, "White Circle: KF Filtered", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("KF Sigma Points Visualization", vis_img)
    cv2.waitKey(1)


if __name__ == "__main__":
    import rclpy

    parser = add_debug_mode(listen_port=9999)
    parser.add_argument("--sim", action="store_true", help="Whether running in simulation mode.")
    parser.add_argument(
        "--frame", type=str, default="camera", help="Frame id for published messages (e.g., 'camera', 'base')."
    )
    parser.add_argument("--num_points", type=int, default=7, help="Number of sigma points to track.")
    args = parser.parse_args()

    cam_cfg = get_realsense_d435i(align_depth=True, compressed=False)

    rclpy.init()
    kf_sigma_node = KFSigmaNode(
        node_name="kf_sigma_node",
        tracker=CartesianVisualOdometryBatchTrackerKF,
        cam_cfg=cam_cfg,
        history_size=100,
        sim_run=args.sim,
        frame_id=args.frame,
        num_points=args.num_points,
    )
    kf_sigma_node.start_spin_standalone()
