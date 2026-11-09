import numpy as np
import quaternion


class CircularBuffer:
    """A circular buffer with fixed length using optimized ring buffer implementation."""

    def __init__(self, length: int):
        self._buffer: np.ndarray | None = None
        self._length = length
        self._num_pushes = 0

    def append(self, value):
        """Append a value to the buffer using ring buffer (optimized, no np.roll).

        Logic matches RL training behavior:
        - First push: Fills entire buffer with 'value'. Next write pointer reset to 0.
        - Subsequent pushes: Overwrites the oldest value (at pointer) and advances pointer.
        """
        if self._length == 0:
            return

        # Initialize buffer on first call
        if self._buffer is None:
            if isinstance(value, np.ndarray):
                self._buffer = np.zeros((self._length,) + value.shape, dtype=value.dtype)
            else:
                self._buffer = np.zeros((self._length, 1), dtype=np.float32)
            self._oldest_idx = 0

        # Special Case: First push fills entire buffer (Reset behavior)
        if self._num_pushes == 0:
            self._buffer[:] = value
            self._oldest_idx = 0  # Next write will overwrite index 0 (the oldest)
        else:
            # Normal Step: Overwrite oldest value
            self._buffer[self._oldest_idx] = value
            self._oldest_idx = (self._oldest_idx + 1) % self._length

        self._num_pushes += 1

    @property
    def buffer(self):
        """Return buffer in chronological order (oldest to newest)."""
        if self._buffer is None:
            return None

        # Optimization: If aligned (ptr=0), return copy directly without concat
        if self._oldest_idx == 0:
            return self._buffer.copy()

        # Ring buffer reconstruction: [oldest...end] + [0...oldest-1]
        # This is required because we treat the buffer as 'full' immediately after first push
        return np.concatenate((self._buffer[self._oldest_idx :], self._buffer[: self._oldest_idx]), axis=0)

    def reset(self):
        if self._buffer is None:
            return
        self._buffer[:] = 0.0
        self._num_pushes = 0
        self._oldest_idx = 0


class VectorLPFilter:
    def __init__(self, sample_period, cutoff_freq, num_channels=3):
        """
        Vector Low Pass Filter (First Order)
        Args:
            sample_period (float): Time between samples (1/sample_freq)
            cutoff_freq (float): Cutoff frequency in Hz. Signals above this freq will be attenuated.
            num_channels (int): Dimension of the vector (e.g. 3 for velocity)
        """
        # Calculate alpha (weight) based on RC filter formula:
        # alpha = dt / (RC + dt)
        # RC = 1 / (2 * pi * fc)
        # alpha = dt / (1/(2*pi*fc) + dt) = (2*pi*fc*dt) / (1 + 2*pi*fc*dt)

        self.weight = (2 * np.pi * sample_period * cutoff_freq) / (1.0 + 2 * np.pi * sample_period * cutoff_freq)
        self.past_values = np.zeros(num_channels)
        self.initialized = False

    def update(self, new_values: np.ndarray):
        if not self.initialized:
            self.past_values = new_values.copy()
            self.initialized = True
        else:
            self.past_values = self.weight * new_values + (1 - self.weight) * self.past_values
        return self.past_values.copy()

    def get_values(self):
        return self.past_values.copy()


def quat_rotate_inverse(q: np.quaternion, v: np.array):
    """q must be numpy-quaternion object in w, x, y, z order
    NOTE: non-batchwise version
    """
    q_inv = q.conjugate()
    return quaternion.rotate_vectors(q_inv, v)


def quat_rotate(q: np.quaternion, v: np.array):
    """q must be numpy-quaternion object in w, x, y, z order
    NOTE: non-batchwise version
    """
    return quaternion.rotate_vectors(q, v)


def warp2pi(angle_rad):
    if angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    elif angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad


def quat_to_rpy_quat(q: np.quaternion):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(2 * (w * y - x * z))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return warp2pi(roll), warp2pi(pitch), warp2pi(yaw)


def quat_to_rpy(quat):
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(2 * (w * y - x * z))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return warp2pi(roll), warp2pi(pitch), warp2pi(yaw)


def rpy_to_quat(euler_rpy):
    roll, pitch, yaw = euler_rpy
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.quaternion(w, x, y, z)


def transform_global_xy_to_robot_xy(global_xy, robot_xy, yaw):
    robot_x = robot_xy[0]
    robot_y = robot_xy[1]
    assert abs(robot_x) < 100 and abs(robot_y) < 100
    global_x = global_xy[0]
    global_y = global_xy[1]
    target_from_go1_xyz = [global_x - robot_x, global_y - robot_y]
    global_x_in_robot = target_from_go1_xyz[0] * np.cos(yaw) + target_from_go1_xyz[1] * np.sin(yaw)
    global_y_in_robot = -target_from_go1_xyz[0] * np.sin(yaw) + target_from_go1_xyz[1] * np.cos(yaw)
    return np.array([global_x_in_robot, global_y_in_robot])


def trans_ros_frame_to_opt_frame(vector_ros):
    """Transform vector from ROS frame to camera frame"""
    # ROS frame: x-forward, y-left, z-up
    # Camera frame: x-right, y-down, z-forward
    R_ros_to_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    vector_cam = R_ros_to_cam @ vector_ros
    return vector_cam


def trans_opt_to_ros_frame(vector_camera):
    """Transform vector from camera frame to ROS frame"""
    # Camera frame: x-right, y-down, z-forward
    # ROS frame: x-forward, y-left, z-up
    R_cam_to_ros = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    vector_ros = R_cam_to_ros @ vector_camera
    return vector_ros


def trans_robot_lin_vel_to_camera_lin_vel(lin_vel_robot, ang_vel_robot, R, T):
    """Transform linear velocity from robot frame to camera frame
    R: rotation from robot to camera (3x3 rotation matrix)
    T: translation from robot to camera (x, y, z)
    lin_vel_robot: linear velocity of robot in robot frame (3,)
    ang_vel_robot: angular velocity of robot in robot frame (3,)
    lin_vel_camera: linear velocity of camera in camera frame (3,)
    """
    # v_cam = R * (v_robot + w_robot x T)
    lin_vel_camera = R @ (lin_vel_robot + np.cross(ang_vel_robot, T))
    return lin_vel_camera


def trans_robot_ang_vel_to_camera_ang_vel(ang_vel_robot, R):
    """Transform angular velocity from robot frame to camera frame
    R: rotation from robot to camera (3x3 rotation matrix)
    ang_vel_robot: angular velocity of robot in robot frame (3,)
    ang_vel_camera: angular velocity of camera in camera frame (3,)
    """
    ang_vel_camera = R @ ang_vel_robot
    return ang_vel_camera
