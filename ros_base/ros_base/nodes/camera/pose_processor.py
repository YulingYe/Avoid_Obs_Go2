import numpy as np
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

from ros_base.configs.camera import CameraProfile


class PoseProcessor:
    def __init__(self, cfg: CameraProfile):
        """
        Args:
            cfg: Configuration object containing camera intrinsics and extrinsics.
        """
        self.img_width = cfg.intrinsics.width
        self.img_height = cfg.intrinsics.height
        self.fx = cfg.intrinsics.fx
        self.fy = cfg.intrinsics.fy
        self.cx = cfg.intrinsics.cx
        self.cy = cfg.intrinsics.cy

        # ------------------------------------------------------------------
        # 1. Pre-compute Extrinsics Matrices (Base <-> Camera)
        # ------------------------------------------------------------------
        # T_B_C: Transform from Base to Camera
        # Assumption: cfg.extrinsics.angles is [roll, pitch, yaw] in degrees.
        # If camera is pitched down 30 deg, pitch should be +30.0 in config.
        self.T_B_C = np.eye(4)
        self.T_B_C[:3, 3] = np.array(cfg.extrinsics.translation)
        self.T_B_C[:3, :3] = R.from_euler("xyz", cfg.extrinsics.angles, degrees=True).as_matrix()

        # T_C_B: Transform from Camera to Base (Inverse of T_B_C)
        # This transforms a point from Camera frame back to Base frame.
        # It inherently includes the "un-pitching" rotation (e.g., -30 deg).
        self.T_C_B = np.linalg.inv(self.T_B_C)

        # ------------------------------------------------------------------
        # 2. Pre-compute World Correction Matrix (World <-> Odom)
        # ------------------------------------------------------------------
        # Assumption: VSLAM initializes the Odom frame aligned with the initial Camera frame.
        # Since the Camera is physically pitched DOWN 30 deg, the Odom frame is also
        # tilted DOWN 30 deg relative to the true Horizon.
        # To align the Odom frame with the horizontal World frame, we must rotate it UP (+30 deg).

        # CORRECTED: Use Positive pitch for correction
        pitch_correction = cfg.extrinsics.angles[1]

        # T_W_O: Transform from Odom (Tilted) to World (Horizontal)
        # This matrix maps a point defined in the Tilted Odom frame to the Horizontal World frame.
        self.T_W_O = np.eye(4)
        # Rotate around Y axis by +Pitch to lift the frame up
        self.T_W_O[:3, :3] = R.from_euler("xyz", [0, pitch_correction, 0], degrees=True).as_matrix()

        # ------------------------------------------------------------------
        # 3. State Variables
        # ------------------------------------------------------------------
        # Current Camera Pose in Odom Frame (Raw VSLAM output)
        self.current_T_O_C = np.eye(4)
        self.is_initialized = False

        # Store the initial absolute world pose to zero out the starting position (Auto-Tare)
        self.initial_T_W_B_abs = None

        # Conversion matrix from Optical Frame to ROS Frame
        self.R_opt_to_ros = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    def update_pose_from_odom(self, msg: Odometry):
        """
        Callback to update the current camera pose from VSLAM odometry message.
        Args:
            msg: nav_msgs/Odometry message (Pose of Camera in Odom Frame)
        """
        # Extract translation
        tx = msg.pose.pose.position.x
        ty = msg.pose.pose.position.y
        tz = msg.pose.pose.position.z

        # Extract rotation (Quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Construct 4x4 Transformation Matrix (T_Odom_Camera)
        T_O_C = np.eye(4)
        T_O_C[:3, 3] = [tx, ty, tz]
        T_O_C[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()

        self.current_T_O_C = T_O_C

        # Initialize the start pose (Auto-Tare)
        if not self.is_initialized:
            # Calculate the absolute world pose of the base at the first frame
            # T_W_B = T_W_O * T_O_C * T_C_B
            initial_pose = self.T_W_O @ self.current_T_O_C @ self.T_C_B
            self.initial_T_W_B_abs = initial_pose
            self.is_initialized = True

    def get_robot_base_world_pose(self):
        """
        Compute Robot Base pose in the corrected Horizontal World Frame.
        The starting position will be (0, 0, 0).

        Returns:
            position: np.array [x, y, z] (Relative to start)
            rpy: np.array [roll, pitch, yaw] (radians, Absolute orientation)
        """
        if not self.is_initialized:
            return np.zeros(3), np.zeros(3)

        # 1. Compute current ABSOLUTE world pose of the base
        # Chain: T_World_Base = T_World_Odom * T_Odom_Camera * T_Camera_Base
        T_W_B_abs = self.T_W_O @ self.current_T_O_C @ self.T_C_B

        # 2. Compute RELATIVE position (Start at 0,0,0)
        # We subtract the initial translation from the current translation.
        # Note: We usually keep absolute orientation (gravity aligned), so we only tare position.
        abs_pos = T_W_B_abs[:3, 3]
        init_pos = self.initial_T_W_B_abs[:3, 3]

        relative_pos = abs_pos - init_pos

        # 3. Extract Rotation (RPY)
        # We use the absolute rotation because T_W_O has already corrected the pitch to be horizontal.
        rpy = R.from_matrix(T_W_B_abs[:3, :3]).as_euler("xyz", degrees=False)

        return relative_pos, rpy

    def get_object_world_position(self, bbox=None, depth=None, point_cam=None):
        """
        Compute object 3D position in the corrected World Frame.
        Args:
            bbox: [xmin, ymin, xmax, ymax] (if using image coordinates)
            depth: float (meters, required if using bbox)
            point_cam: list or np.array [x, y, z] (Optional, if already in camera optical frame)
        Returns:
            p_world: np.array [x, y, z] (Relative to robot start position)
        """
        if not self.is_initialized:
            # print("Warning: Pose not initialized yet.")
            return np.zeros(3)

        if point_cam is not None:
            # Point in Camera Optical Frame [x, y, z, 1]
            P_cam_optical = np.eye(4)[:1]  # dummy for size
            P_cam_optical = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
        else:
            # 1. Compute UV center
            u = (bbox[0] + bbox[2]) / 2.0
            v = (bbox[1] + bbox[3]) / 2.0

            # 2. Back-project to Camera Frame (P_cam)
            # Formula: x = (u - cx) * z / fx
            # Note: In standard optical frame: X-right, Y-down, Z-forward.
            # BUT your extrinsics/odom might use ROS frame conventions.
            # Standard Realsense SDK projection gives Optical Frame coordinates.
            z_c = depth
            x_c = (u - self.cx) * z_c / self.fx
            y_c = (v - self.cy) * z_c / self.fy

            # Point in Camera Optical Frame [x, y, z, 1]
            P_cam_optical = np.array([x_c, y_c, z_c, 1.0])

        # CRITICAL: If your T_C_B / T_O_C are based on ROS Frame (X-forward),
        # but projection gives Optical Frame (Z-forward), we need a conversion here.
        # Assuming VSLAM output and Extrinsics are consistently using ROS Frame convention
        # (as you mentioned odom is ROS frame), we need to rotate P_cam_optical to P_cam_ros.

        # Optical (Z-fwd, Y-down) -> ROS (X-fwd, Z-up)
        # R = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        P_cam_ros = self.R_opt_to_ros @ P_cam_optical

        # 3. Transform to World Frame
        # Chain: P_World_Abs = T_World_Odom * T_Odom_Camera * P_Camera_ROS

        # Step A: Camera -> Odom
        P_odom = self.current_T_O_C @ P_cam_ros

        # Step B: Odom -> World (Absolute)
        P_world_abs = self.T_W_O @ P_odom

        # Step C: Subtract initial position to match the robot's relative frame
        # P_world_rel = P_world_abs - P_init
        P_world_rel = P_world_abs[:3] - self.initial_T_W_B_abs[:3, 3]

        return P_world_rel
