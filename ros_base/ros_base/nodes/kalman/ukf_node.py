import cv2
import numpy as np
from geometry_msgs.msg import PointStamped
from kalman.simulation.base_scene import CameraParameters
from kalman.tracker.cartesian.visual_odometry_ukf import (
    CartesianVisualOdometryTrackerUKF,
)
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile

from ros_base.configs.camera import get_realsense_d435i
from ros_base.nodes.base_node import BaseNode
from ros_base.utils.args_debug import add_debug_mode
from ros_base.utils.camera_trans import CameraTrans
from ros_base.utils.decorators import profile_latency
from ros_base.utils.math_utils import *

INVALID_MEAS = -1.0


class UKFNode(BaseNode):
    def __init__(
        self,
        tracker: object = CartesianVisualOdometryTrackerUKF,
        cam_cfg=None,
        history_size: int = 100,
        sim_run: bool = False,
        frame_id: str = "camera",
        max_lost_time: float = 2.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        camera_params = CameraParameters(
            focal_length=(cam_cfg.intrinsics.fx, cam_cfg.intrinsics.fy),
            principal_point=(cam_cfg.intrinsics.cx, cam_cfg.intrinsics.cy),
            image_size=(cam_cfg.intrinsics.width, cam_cfg.intrinsics.height),
            distortion_coeffs=getattr(cam_cfg.intrinsics, "D", None),
        )

        self.tracker: CartesianVisualOdometryTrackerUKF = self.create_tracker(tracker, camera_params, history_size)
        self.sim_run = sim_run
        self.frame_id = frame_id
        self.max_lost_time = max_lost_time
        self.last_valid_time = 0.0
        self.camera_trans = CameraTrans(cfg=cam_cfg)

        self.p_img_sub = self.create_subscription(PointStamped, "/geometry_msgs/p_img", self._perception_callback, 1)

        self.p_img_filter_pub = self.create_publisher(PointStamped, "/geometry_msgs/p_img_filtered", 1)
        self.p_img_filter_msg = PointStamped()
        self.create_timer(0.02, lambda: self.publish_p_img_filter(vis=self.sim_run))  # 50 Hz

        # qos_odom = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.odom_sub = self.create_subscription(
            Odometry,
            "/visual_slam/tracking/odometry",
            self._pose_callback,
            1,
        )

        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_quat_wxyz = np.quaternion(1, 0, 0, 0)

        if self.sim_run:
            self.p_img_truth_sub = self.create_subscription(
                PointStamped, "/geometry_msgs/p_img_truth", self._p_img_truth_callback, 1
            )  # used for simulation only
            self.p_img_truth = np.array([INVALID_MEAS, INVALID_MEAS, INVALID_MEAS])

            self.pose_sub = self.create_subscription(
                Odometry, "/mujoco/odom", self._pose_callback, 10
            )  # robot base odom

            self.frame_id = "base"  # In sim, odom is in robot base frame

        # Precompute transformation matrices to reduce latency in callbacks
        # T_bc = [self.camera_trans.R.T, self.camera_trans.T; 0, 1]
        self.T_bc = np.eye(4)
        self.T_bc[:3, :3] = self.camera_trans.R.T
        self.T_bc[:3, 3] = self.camera_trans.T

        # R_ros_to_opt columns are axes of Optical frame in ROS frame
        R_ros_to_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        self.T_ros_to_opt = np.eye(4)
        self.T_ros_to_opt[:3, :3] = R_ros_to_opt

        self.det_lag_pos = np.array([INVALID_MEAS, INVALID_MEAS, INVALID_MEAS])

    def create_tracker(self, tracker_class, camera_params, history_size):
        # Create tracker instance
        tracker = tracker_class(
            camera_params, history_size=history_size, dt=0.02
        )  # Default dt, will be overwritten by predict
        return tracker

    def _get_valid_data(self, msg_x, msg_y, msg_z):
        """Check measurement validity, denormalize u, v"""
        validity = (msg_x != INVALID_MEAS) and (msg_y != INVALID_MEAS)
        if not validity:
            self.det_lag_pos = np.array([INVALID_MEAS, INVALID_MEAS, INVALID_MEAS])
            return validity

        z_val = msg_z if msg_z != INVALID_MEAS else INVALID_MEAS
        u, v, z = msg_x * self.tracker.camera.width, msg_y * self.tracker.camera.height, z_val
        self.det_lag_pos = np.array([u, v, z])
        return validity

    def _handle_validity(self, valid, timestamp):
        """Handle tracker validity state"""
        if not valid:
            # Target out of view, stop tracker
            if self.tracker.initialized:
                if (timestamp - self.last_valid_time) > self.max_lost_time:
                    self.tracker.reset()
                    self.logger.info(f"Target lost for > {self.max_lost_time}s. Resetting tracker.")
        else:
            self.last_valid_time = timestamp
            if not self.tracker.initialized:
                # Initialize tracker
                self.tracker.initialize(
                    u_init=self.det_lag_pos[0],
                    v_init=self.det_lag_pos[1],
                    z_init=self.det_lag_pos[2],
                    u_imu_init=np.zeros(6),
                    timestamp=timestamp,
                )
                self.logger.info("UKF Tracker initialized.")

    def _compute_timestamp(self):
        # Transform to float seconds
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    @profile_latency(
        cycle_threshold_ms=250.0,
        process_threshold_ms=20.0,
        log_interval_s=10.0,
        debug=True,
        check_capture_latency=False,
    )
    def _pose_callback(self, msg: Odometry):
        """Transform robot base odom to camera frame odom and feed to UKF"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self.tracker.initialized:
            return

        if hasattr(self.tracker.ukf, "last_timestamp") and timestamp <= self.tracker.ukf.last_timestamp:
            self.logger.warning(f"Odom timestamp {timestamp} <= last {self.tracker.ukf.last_timestamp}, skipping.")
            return

        # 1. Get Pose from Odometry msg
        # Position
        tx = msg.pose.pose.position.x
        ty = msg.pose.pose.position.y
        tz = msg.pose.pose.position.z
        # Orientation (Quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Optimized quaternion to rotation matrix (avoid scipy overhead)
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        xw = qx * qw
        yw = qy * qw
        zw = qz * qw

        pos_odom = np.array([tx, ty, tz])  # in optical frame
        self.camera_quat_wxyz.w = qw
        self.camera_quat_wxyz.x = qx
        self.camera_quat_wxyz.y = qy
        self.camera_quat_wxyz.z = qz

        # Rotation matrix in Optical Frame (from Odometry)
        R_odom = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
                [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
                [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)],
            ]
        )

        # self.camera_pos = trans_opt_to_ros_frame(pos_odom)  # in ROS frame
        # Transform rotation matrix to ROS frame for external usage
        # R_ros = R_opt_to_ros @ R_odom
        # self.camera_rot_mat = self.T_ros_to_opt[:3, :3] @ R_odom

        T_msg = np.eye(4)
        T_msg[:3, :3] = R_odom
        T_msg[:3, 3] = pos_odom

        # Check if the odometry is already in camera frame
        if self.frame_id == "camera":
            T_wc = T_msg @ self.T_ros_to_opt
        else:
            T_wc = T_msg @ self.T_bc

        # 4. Predict
        self.tracker.predict(current_pose_wc=T_wc, timestamp=timestamp)

    @profile_latency(
        cycle_threshold_ms=250.0,
        process_threshold_ms=30.0,
        log_interval_s=10.0,
        debug=True,
        check_capture_latency=False,
    )
    def _perception_callback(self, msg: PointStamped):
        # Use the timestamp from the message header
        capture_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        valid = self._get_valid_data(msg.point.x, msg.point.y, msg.point.z)  # Denormalize u, v, and get validity
        self._handle_validity(valid, capture_time)

        if valid:
            # Check for divergence before update
            if self.tracker.initialized:
                state = self.tracker.get_state()
                pred_u, pred_v, pred_z = state["position"]
                meas_u, meas_v, meas_z = self.det_lag_pos

                # Divergence Check:
                # 1. Z collapsed to min (0.1) while measurement is far (> 0.3)
                # 2. Pixel error is huge (> 400 px)
                z_diverged = (pred_z < 0.2) and (meas_z > 0.3)
                uv_diverged = (abs(pred_u - meas_u) > 400) or (abs(pred_v - meas_v) > 400)

                if z_diverged or uv_diverged:
                    self.logger.warning(
                        f"UKF Diverged (Z_pred={pred_z:.2f}, Z_meas={meas_z:.2f}, "
                        f"du={abs(pred_u - meas_u):.1f}, dv={abs(pred_v - meas_v):.1f}). Resetting."
                    )
                    self.tracker.reset()
                    self.tracker.initialize(
                        u_init=meas_u,
                        v_init=meas_v,
                        z_init=meas_z,
                        u_imu_init=np.zeros(6),
                        timestamp=capture_time,
                    )
                    return

            # Update tracker
            if self.tracker.dim_z == 3:
                cam_data = self.det_lag_pos
            else:  # Without depth measurement
                cam_data = self.det_lag_pos[0:2]

            try:
                self.tracker.update(timestamp=capture_time, cam_data=cam_data, debug=False)
            except ValueError as e:
                self.logger.warn(f"Tracker update failed: {e}")

    def publish_p_img_filter(self, vis=False):
        est_pos = self._get_est_pos()

        if self.tracker.initialized:
            x_filter = est_pos[0] / self.tracker.camera.width
            y_filter = est_pos[1] / self.tracker.camera.height
            z_filter = est_pos[2]
        else:
            x_filter = INVALID_MEAS
            y_filter = INVALID_MEAS
            z_filter = INVALID_MEAS

        self.p_img_filter_msg.header.stamp = self.get_clock().now().to_msg()
        self.p_img_filter_msg.header.frame_id = "camera_frame"
        self.p_img_filter_msg.point.x = float(x_filter)
        self.p_img_filter_msg.point.y = float(y_filter)
        self.p_img_filter_msg.point.z = float(z_filter)
        self.p_img_filter_pub.publish(self.p_img_filter_msg)

        if vis:
            visualize_p_img(
                est_pos=self.est_pos,
                gd_pos=self.det_lag_pos,
                width=self.tracker.camera.width,
                height=self.tracker.camera.height,
                p_img_truth=self.p_img_truth,
            )

    def _get_est_pos(self):
        # valid = (self.det_lag_pos[0] != INVALID_MEAS) and (self.det_lag_pos[1] != INVALID_MEAS)
        if not self.tracker.initialized:
            self.est_pos = np.array([INVALID_MEAS, INVALID_MEAS, INVALID_MEAS])
            return self.est_pos
        state = self.tracker.get_state()
        est_pos = state["position"]
        u_est = est_pos[0]
        v_est = est_pos[1]
        z_est = est_pos[2]
        # z_est = z_est if valid else INVALID_MEAS  # when invalid uv, set z to -1.0 as well
        self.est_pos = np.array([u_est, v_est, z_est])
        return self.est_pos

    def _p_img_truth_callback(self, msg: PointStamped):
        """Callback to get ground truth p_img data (get from simulator)"""
        validity = (msg.point.x != INVALID_MEAS) and (msg.point.y != INVALID_MEAS) and (msg.point.z != INVALID_MEAS)
        if not validity:
            self.p_img_truth = np.array([INVALID_MEAS, INVALID_MEAS, INVALID_MEAS])
            return
        u, v, z = msg.point.x * self.tracker.camera.width, msg.point.y * self.tracker.camera.height, msg.point.z
        self.p_img_truth = np.array([u, v, z])


def visualize_p_img(est_pos, gd_pos, width, height, p_img_truth=None):
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw ground truth position in green
    if gd_pos is not None and (gd_pos[0] != INVALID_MEAS and gd_pos[1] != INVALID_MEAS):
        cv2.circle(vis_img, (int(gd_pos[0]), int(gd_pos[1])), 5, (0, 255, 0), -1)
        # log info on the image
        cv2.putText(
            vis_img,
            f"p_img: ({gd_pos[0]:.1f}, {gd_pos[1]:.1f}, {gd_pos[2]:.2f}m)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    # Draw estimated position in white
    if est_pos is not None and (est_pos[0] != INVALID_MEAS and est_pos[1] != INVALID_MEAS):
        cv2.circle(vis_img, (int(est_pos[0]), int(est_pos[1])), 5, (255, 255, 255), -1)
        # log info on the image
        cv2.putText(
            vis_img,
            f"p_img_filt: ({est_pos[0]:.1f}, {est_pos[1]:.1f}, {est_pos[2]:.2f}m)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    # Draw truth position in red
    if p_img_truth is not None and (p_img_truth[0] != INVALID_MEAS and p_img_truth[1] != INVALID_MEAS):
        cv2.circle(vis_img, (int(p_img_truth[0]), int(p_img_truth[1])), 5, (0, 0, 255), -1)

    cv2.imshow("P Image Visualization", vis_img)
    cv2.waitKey(1)


if __name__ == "__main__":
    import rclpy

    parser = add_debug_mode(listen_port=7777)
    parser.add_argument("--sim", action="store_true", help="Whether running in simulation mode.")
    parser.add_argument(
        "--frame", type=str, default="camera", help="Frame id for published messages (e.g., 'camera', 'base')."
    )
    parser.add_argument(
        "--max_lost_time", type=float, default=2.0, help="Maximum time (s) to keep lost target before resetting UKF."
    )
    args = parser.parse_args()

    cam_cfg = get_realsense_d435i(align_depth=True, compressed=False)  # Use Realsense camera config

    rclpy.init()
    ukf_node = UKFNode(
        node_name="ukf_node",
        tracker=CartesianVisualOdometryTrackerUKF,
        cam_cfg=cam_cfg,
        history_size=100,
        sim_run=args.sim,
        frame_id=args.frame,
        max_lost_time=args.max_lost_time,
    )
    ukf_node.start_spin_standalone()
