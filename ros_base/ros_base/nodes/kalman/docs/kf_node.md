# KFNode Documentation

`KFNode` is a ROS 2 node that implements a Linear Kalman Filter for 3D target tracking in the camera frame. It serves as a robust alternative to the `UKFNode`, specifically designed to handle visual odometry tracking with delayed measurements and camera motion compensation.

## Node Description

**File**: `ros_base/nodes/kalman/kf_node.py`

The node subscribes to target detections in the image plane (`p_img`) and robot odometry (`odom`). It fuses these inputs to estimate the 3D position and velocity of the target relative to the camera, compensating for the robot's ego-motion.

## Key Features

1.  **Linear Kalman Filter**: Uses `CartesianVisualOdometryTrackerKF` which linearizes the measurement model via back-projection, avoiding stability issues associated with UKF sigma points at close range.
2.  **Asynchronous Updates**: Handles the time delay between image capture and processing by using a rollback-and-replay mechanism.
3.  **Motion Compensation**: Explicitly accounts for camera movement using high-frequency odometry data, allowing for accurate tracking even when the robot is moving fast.
4.  **Blind Spot Prediction**: Includes a `max_lost_time` parameter. When the target is lost (e.g., enters a blind spot below the camera), the filter continues to predict the target's position based on the last known state and camera motion for a specified duration.

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sim` | bool | `False` | Whether running in simulation mode (subscribes to ground truth topics). |
| `frame` | string | `"base"` | The reference frame ID for published messages. |
| `max_lost_time` | float | `2.0` | Maximum time (in seconds) to keep predicting the target state after tracking is lost before resetting the filter. |

## Subscribed Topics

*   `/geometry_msgs/p_img` (`geometry_msgs/PointStamped`):
    *   Target detection in normalized image coordinates (or pixel coordinates depending on upstream).
    *   Format: `x`, `y` are pixel/normalized coordinates, `z` is depth.
*   `/visual_slam/tracking/odometry` (`nav_msgs/Odometry`):
    *   Robot/Camera odometry used for motion compensation.
*   `/geometry_msgs/p_img_truth` (`geometry_msgs/PointStamped`) [Sim only]:
    *   Ground truth target position for visualization/debugging.

## Published Topics

*   `/geometry_msgs/p_img_filtered` (`geometry_msgs/PointStamped`):
    *   The filtered 3D position of the target in the camera frame.
    *   Published at a fixed rate (50Hz).

## Logic Flow

1.  **Initialization**: The tracker waits for the first valid detection (`p_img`) to initialize the state vector $[X, Y, Z, 0, 0, 0]$.
2.  **Prediction Loop (Odom Callback)**:
    *   Receives high-frequency odometry.
    *   Calculates the relative transform of the camera since the last step.
    *   Calls `tracker.predict()` to transform the estimated target state into the new camera frame.
3.  **Correction Loop (Perception Callback)**:
    *   Receives a detection with a timestamp $t_{capture}$.
    *   Checks validity. If valid, calls `tracker.update()`.
    *   If invalid (target lost), checks if `time_since_last_valid < max_lost_time`. If so, maintains the tracker (relying on prediction). If not, resets the tracker.
4.  **Publishing**: A timer triggers `publish_p_img_filter` to broadcast the current best estimate of the target position.

## Differences from UKFNode

*   **Algorithm**: Uses Standard KF instead of Unscented KF.
*   **Stability**: More robust to depth singularities ($Z \to 0$) and large initial errors.
*   **Divergence Check**: Removed the explicit divergence check logic found in `UKFNode` as the linear formulation is inherently less prone to the specific "sigma point explosion" instability.
