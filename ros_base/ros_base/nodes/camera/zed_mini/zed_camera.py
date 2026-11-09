import time

import cv2
import numpy as np
import pyzed.sl as sl

from ros_base.utils.args_debug import add_debug_mode
from ros_base.utils.math_utils import *


class ZedCamera:
    # Resolution mode mapping
    RESOLUTION_MODES = {"HD720": sl.RESOLUTION.HD720, "VGA": sl.RESOLUTION.VGA}  # 1280x720  # 672x376

    # Depth mode mapping
    DEPTH_MODES = {
        "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
        "NEURAL": sl.DEPTH_MODE.NEURAL,
        "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
    }

    def __init__(self, resolution_mode="HD720", depth_mode="NEURAL", enable_imu=False, scale: int = 2, *args, **kwargs):
        """Class to interface with Zed Mini camera using pyzed.sl SDK.
        Args:
            resolution_mode (str): Resolution mode for the camera ("HD720" or "VGA").
            depth_mode (str): Depth mode for the camera ("NEURAL_LIGHT", "NEURAL", or "NEURAL_PLUS").
            scale (int): Scale factor for resizing the image.
                - scale = 1: original resolution (1280x720)
                - scale = 2: half resolution (640x360)
                - scale = 4: quarter resolution (320x180)
                - scale = 8: eighth resolution (160x90)
        """
        self.zed = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.K = None  # Camera intrinsics matrix
        self.HFOV = None  # Camera HFOV
        self.VFOV = None  # Camera VFOV
        self.DFOV = None  # Camera DFOV
        self.rpy = None  # Euler angles
        self.T = None  # Translation vector, camera optical center position in base coordinate
        self.camera_resolution = None  # Camera resolution setting
        self.camera_depth_mode = None  # Camera depth mode setting
        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.image_depth = sl.Mat()
        self.current_image = None
        self.current_point_cloud = None
        self.image_width = None
        self.image_height = None
        self.image_depth_width = None
        self.image_depth_height = None
        self.point_cloud_width = None
        self.point_cloud_height = None
        self.depth_max = 10000.0  # mm, 10 meters
        self.depth_min = 100.0  # mm, 0.1 meters
        self.scale = scale  # scale down the image for faster processing
        self.initialize_camera(resolution_mode, depth_mode, enable_imu)
        self.get_intrinsics()

    def initialize_camera(self, resolution_mode, depth_mode, enable_imu):
        """Initialize ZED camera and open the device."""
        print("Initializing ZED camera...")

        # Validate and set resolution mode
        if resolution_mode not in ZedCamera.RESOLUTION_MODES:
            raise ValueError(
                f"Invalid resolution_mode: {resolution_mode}. Available options:"
                f" {list(ZedCamera.RESOLUTION_MODES.keys())}"
            )
        self.camera_resolution = ZedCamera.RESOLUTION_MODES[resolution_mode]
        print(f"Resolution mode set to: {resolution_mode}")

        # Validate and set depth mode
        if depth_mode not in ZedCamera.DEPTH_MODES:
            raise ValueError(
                f"Invalid depth_mode: {depth_mode}. Available options: {list(ZedCamera.DEPTH_MODES.keys())}"
            )
        self.camera_depth_mode = ZedCamera.DEPTH_MODES[depth_mode]
        print(f"Depth mode set to: {depth_mode}")

        init_params = sl.InitParameters()
        init_params.camera_resolution = self.camera_resolution
        init_params.depth_mode = self.camera_depth_mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # sl.UNIT.METER

        self.zed = sl.Camera()
        self.zed.close()

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open the camera: {repr(status)}")
            self.zed = None
            return False
        print("ZED camera initialized successfully.")

        if enable_imu:
            self.initialize_pose()

        return True

    def get_intrinsics(self):
        """
        Get intrinsics matrix K (fx, fy, cx, cy) from ZED camera.
        """
        if self.zed is None:
            print("Error: Camera not initialized, unable to get intrinsics.")
            return False

        print("Getting camera intrinsics...")
        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters

        self.fx = calib.left_cam.fx
        self.fy = calib.left_cam.fy
        self.cx = calib.left_cam.cx
        self.cy = calib.left_cam.cy
        self.HFOV = calib.left_cam.h_fov
        self.VFOV = calib.left_cam.v_fov
        self.DFOV = calib.left_cam.d_fov
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        print(f"-" * 40)
        print("Camera Intrinsics:")
        print("fx:", self.fx)
        print("fy:", self.fy)
        print("cx:", self.cx)
        print("cy:", self.cy)
        print("HFOV (degrees):", self.HFOV)
        print("VFOV (degrees):", self.VFOV)
        print("DFOV (degrees):", self.DFOV)
        print(f"-" * 40)
        return True

    def set_extrinsics(self, rpy, T):
        """
        Manually set extrinsics: Euler angles rpy and translation vector T
        """
        self.rpy = np.array(rpy, dtype=np.float64)
        self.T = np.array(T, dtype=np.float64)

        print("Extrinsics setting completed.")
        print("Euler angles rpy:\n", self.rpy)
        print("Translation vector T:\n", self.T)
        return True

    def capture_image(self, grab=True, *args, **kwargs):
        """
        Capture a frame, and save the color image.
        Args:
            grab (bool): Whether to grab a new frame before retrieving image.
        Returns:
            bool: True if successful, False otherwise.
            np.ndarray: Color image in BGR format.
        """
        err = self.zed.grab(self.runtime_parameters) if grab else sl.ERROR_CODE.SUCCESS
        if err > sl.ERROR_CODE.SUCCESS:  # Check that a new image is successfully acquired
            print(f"Error: Failed to capture a frame: {repr(err)}")
            return False, None

        self.zed.retrieve_image(
            self.image, sl.VIEW.LEFT, sl.MEM.CPU, sl.Resolution(1280 // self.scale, 720 // self.scale)
        )
        raw_image = self.image.get_data()  # BGRA format
        self.current_image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)  # BGR format
        if self.image_width is None or self.image_height is None:
            self.image_width = self.image.get_width()
            self.image_height = self.image.get_height()
            print(f"Image resolution: {self.image_width} x {self.image_height}")
        return True, self.current_image

    def capture_depth(self, grab=True, *args, **kwargs):
        """
        Capture a frame, and save the depth data.
        Args:
            grab (bool): Whether to grab a new frame before retrieving depth.
        Returns:
            bool: True if successful, False otherwise.
            np.ndarray: Depth image in mm (single channel uint16).
        """
        err = self.zed.grab(self.runtime_parameters) if grab else sl.ERROR_CODE.SUCCESS
        if err > sl.ERROR_CODE.SUCCESS:  # Check that a new image is successfully acquired
            print(f"Error: Failed to capture a frame: {repr(err)}")
            return False, None

        self.zed.retrieve_measure(
            self.image_depth, sl.MEASURE.DEPTH_U16_MM, sl.MEM.CPU, sl.Resolution(1280 // self.scale, 720 // self.scale)
        )
        current_depth = self.image_depth.get_data()  # single channel uint16, depth in mm
        if self.image_depth_width is None or self.image_depth_height is None:
            self.image_depth_width = self.image_depth.get_width()
            self.image_depth_height = self.image_depth.get_height()
            print(f"image_depth resolution: {self.image_depth_width} x {self.image_depth_height}")
        return True, current_depth

    def capture_point_cloud(self, grab=True, *args, **kwargs):
        """
        Capture a frame, and save the point cloud data.
        Args:
            grab (bool): Whether to grab a new frame before retrieving point cloud.
        Returns:
            bool: True if successful, False otherwise.
            np.ndarray: Point cloud data (3 channels: X, Y, Z in mm).
        """
        err = self.zed.grab(self.runtime_parameters) if grab else sl.ERROR_CODE.SUCCESS
        if err > sl.ERROR_CODE.SUCCESS:  # Check that a new image is successfully acquired
            print(f"Error: Failed to capture a frame: {repr(err)}")
            return False, None

        # Get depth point cloud
        self.zed.retrieve_measure(
            self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, sl.Resolution(1280 // self.scale, 720 // self.scale)
        )
        current_point_cloud = self.point_cloud.get_data()
        if self.point_cloud_width is None or self.point_cloud_height is None:
            self.point_cloud_width = self.point_cloud.get_width()
            self.point_cloud_height = self.point_cloud.get_height()
            print(f"point_cloud resolution: {self.point_cloud_width} x {self.point_cloud_height}")
        return True, current_point_cloud

    def initialize_pose(self):
        """Initialize the pose estimation of the camera.
        Returns:
            bool: True if successful, False otherwise.
        """
        tracking_params = sl.PositionalTrackingParameters()  # set parameters for Positional Tracking
        tracking_params.enable_imu_fusion = True
        tracking_params.enable_area_memory = False
        # tracking_params.set_floor_as_origin = True # take some time to find the floor plane
        tracking_params.enable_pose_smoothing = True
        tracking_params.set_gravity_as_origin = False  # False: set initial pose as origin (0,0,0,0,0,0), True: use world frame (gravity aligned), initial pose: (0,0,0, r0,p0,y0)
        # False: IMU frame is aligned with camera frame at the initial time
        # True: IMU frame is aligned with world frame (gravity aligned) at the initial time
        # But our IMU sensor cannot access.
        status = self.zed.enable_positional_tracking(tracking_params)  # enable Positional Tracking
        if status != sl.ERROR_CODE.SUCCESS:
            print("Enable Positional Tracking : " + repr(status) + ". Exit program.")
            self.zed.close()
            raise Exception("Failed to enable positional tracking.")

        self.camera_pose = sl.Pose()
        self.py_translation = sl.Translation()
        self.py_orientation = sl.Orientation()
        self.sensor_data = sl.SensorsData()
        self.last_translation = None
        self.last_timestamp = None

        return True

    def get_orientation(self):
        """Get the current orientation quaternion of the camera.
        Returns:
            np.quaternion: The current orientation quaternion.
        """
        ox = self.camera_pose.get_orientation(self.py_orientation).get()[0]
        oy = self.camera_pose.get_orientation(self.py_orientation).get()[1]
        oz = self.camera_pose.get_orientation(self.py_orientation).get()[2]
        ow = self.camera_pose.get_orientation(self.py_orientation).get()[3]
        return np.quaternion(ow, ox, oy, oz)

    def get_euler_angles(self):
        """Get the current euler angles (roll, pitch, yaw) of the camera.
        Returns:
            np.array: Euler angles [roll, pitch, yaw] in radians.
        """
        euler_rpy = self.camera_pose.get_euler_angles()
        return np.array(euler_rpy)

    def get_translation(self):
        """Get the current translation vector (x, y, z) of the camera.
        Returns:
            np.array: Translation vector [x, y, z] in mm.
        """
        translation = self.camera_pose.get_translation(self.py_translation)
        return np.array([translation.get()[0], translation.get()[1], translation.get()[2]])

    def get_velocity(self):
        """Get the current velocity of the camera.
        Returns:
            np.array: Linear velocity vector [vx, vy, vz] in m/s.
        """
        tracking_state = self.zed.get_position(
            self.camera_pose, sl.REFERENCE_FRAME.WORLD
        )  # Get the position of the camera in a fixed reference frame (the World Frame)
        if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
            translation = self.get_translation() / 1e3  # convert mm to m
            quaternion = self.get_orientation()
            euler_rpy = self.get_euler_angles()
            # quat_ = rpy_to_quat(euler_rpy)
            timestamp = self.camera_pose.timestamp.get_nanoseconds()
            if self.last_timestamp is None:
                self.last_translation = translation
                self.last_euler_rpy = euler_rpy
                self.last_timestamp = timestamp
                lin_vel_camera = np.array([0.0, 0.0, 0.0])
                ang_vel_camera = np.array([0.0, 0.0, 0.0])
            else:
                lin_vel_world = (translation - self.last_translation) / ((timestamp - self.last_timestamp) / 1e9)
                lin_vel_camera = quat_rotate_inverse(quaternion, lin_vel_world)
                ang_vel_camera = (euler_rpy - self.last_euler_rpy) / ((timestamp - self.last_timestamp) / 1e9)
                self.last_translation = translation
                self.last_euler_rpy = euler_rpy
                self.last_timestamp = timestamp
            return True, lin_vel_camera, ang_vel_camera
        else:
            print(f"Warning: Tracking state not OK: {repr(tracking_state)}")
            return False, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

    def get_pose(self):
        """Get the current pose of the camera.
        Returns:
            bool: True if successful, False otherwise.
            dict: Pose data containing 'position', 'orientation', and 'timestamp'.
        """
        tracking_state = self.zed.get_position(
            self.camera_pose, sl.REFERENCE_FRAME.WORLD
        )  # Get the position of the camera in a fixed reference frame (the World Frame)
        if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
            # quaternion = self.get_orientation()
            translation = self.get_translation() / 1e3  # convert mm to m
            euler_rpy = self.get_euler_angles()
            timestamp = self.camera_pose.timestamp.get_nanoseconds()
            pose_data = {
                "position": translation,  # in m
                "euler_rpy": euler_rpy,  # in radians
                "timestamp": timestamp / 1e9,  # in seconds
            }
            return True, pose_data
        else:
            print(f"Warning: Tracking state not OK: {repr(tracking_state)}")
            return False, None

    def close(self):
        """close the camera and release the resources."""
        if self.zed is not None:
            self.zed.close()
            self.zed = None
            print("ZED camera closed.")


def display_cam_pose(cam_pose, lin_vel=None, ang_vel=None):
    """Display the camera pose data in a readable format.
    Args:
        pose_data (dict): Pose data containing 'position', 'orientation', 'velocity', and 'timestamp'.
    """
    if cam_pose is not None:
        position = cam_pose.get("position", None)  # xyz is in camera frame
        # orientation = cam_pose.get("orientation", None)
        roll, pitch, yaw = euler_rpy
        euler_rpy = cam_pose.get("euler_rpy", None)
        timestamp = cam_pose.get("timestamp", None)
        print(f"Camera Position (m): x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
        print(f"Camera Orientation (rad): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
        print(f"Camera Linear Velocity (m/s): vx={lin_vel[0]:.2f}, vy={lin_vel[1]:.2f}, vz={lin_vel[2]:.2f}")
        print(f"Camera Angular Velocity (rad/s): wx={ang_vel[0]:.3f}, wy={ang_vel[1]:.3f}, wz={ang_vel[2]:.3f}")
        print(f"Timestamp (s): {timestamp}")
    else:
        print("No valid pose data available for visualization.")


def track_camera_pose_example(camera: ZedCamera):
    pose_start = time.monotonic()
    NUM_ITR = 1000

    for i in range(NUM_ITR):
        camera.capture_image()  # Important: must capture image to update pose
        ret_pose, cam_pose = camera.get_pose()
        # lin_vel = camera.get_linear_velocity()
        # ang_vel = camera.get_angular_velocity()
        ret_vel, lin_vel, ang_vel = camera.get_velocity()
        if ret_pose and ret_vel:
            if i % 10 == 0:
                print(f"Step_{i}:")
                display_cam_pose(cam_pose, lin_vel, ang_vel)
        else:
            print(f"Step_{i}: Failed to get camera pose.")

    pose_end = time.monotonic()
    pose_time = pose_end - pose_start
    pose_time /= NUM_ITR

    # print result
    print(f"Pose query time: {pose_time*1e3:.3f} ms")


def display_rgb_depth_image(camera: ZedCamera):
    depth_start = time.monotonic()
    NUM_ITR = 100
    from ros_base.utils.img_utils import get_depth_at_pixel_depth

    # only show image
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    for i in range(NUM_ITR):
        _, current_image = camera.capture_image()
        _, current_depth = camera.capture_depth()
        depth = get_depth_at_pixel_depth(current_depth=current_depth, u=320, v=180)
        cv2.imshow("image", current_image)
        cv2.waitKey(1)
        print(f"Depth at step_{i}: {depth/1000:.2f} m")

    depth_end = time.monotonic()
    depth_time = depth_end - depth_start
    depth_time /= NUM_ITR

    # print result
    print(f"Depth query time: {depth_time*1e3:.3f} ms")


if __name__ == "__main__":
    parser = add_debug_mode(listen_port=9999)  # 9999: wireless debug port, 7777: wired debug port
    parser.add_argument("--enable_imu", default=True, help="Enable IMU data retrieval if supported by the camera.")
    args = parser.parse_args()

    camera = ZedCamera("VGA", "NEURAL", enable_imu=args.enable_imu)

    display_rgb_depth_image(camera)
    track_camera_pose_example(camera)

    camera.close()
