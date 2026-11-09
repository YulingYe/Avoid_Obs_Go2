"""Local Unitree MuJoCo backend with SEA-Nav ROS 2 perception publishers.

Run from any directory with ``python -m quad_deploy.scripts.sea.mujoco_sim``.
The upstream simulator is imported only for its SDK bridge, without launching
its viewer or requiring a gamepad. Pose is simulation ground truth. Rays are
horizontal (yaw-only), measured at the base origin plus lidar_height.
"""

import argparse
import importlib.util
import math
from pathlib import Path
import threading
import time
from contextlib import nullcontext

import mujoco
import numpy as np


class Perception:
    """Sample one consistent physics state; exclude the entire robot subtree."""

    def __init__(self, model, base_body="base_link", lidar_height=0.05):
        self.model = model
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        if self.base_id < 1:
            raise ValueError("Robot base body not found: " + base_body)
        self.lidar_height = lidar_height
        self.angles = np.linspace(-2 * math.pi / 3, 2 * math.pi / 3, 41)
        self.range_min, self.range_max = 0.1, 3.0
        robot_bodies = {self.base_id}
        for body_id in range(self.base_id + 1, model.nbody):
            if int(model.body_parentid[body_id]) in robot_bodies:
                robot_bodies.add(body_id)
        # Reserve group 5 for robot geometry. Remap existing scene group 5 to
        # group 0; groups affect visibility/ray selection, not collision forces.
        self.ray_groups = model.geom_group.copy()
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) in robot_bodies:
                self.ray_groups[geom] = 5
            elif model.geom_group[geom] == 5:
                self.ray_groups[geom] = 0
        self.groups = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)

    def sample(self, data):
        position = data.xpos[self.base_id].copy()
        rotation = data.xmat[self.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        origin = position.copy()
        origin[2] += self.lidar_height
        ranges = []
        hit = np.empty(1, dtype=np.int32)
        original_groups = self.model.geom_group.copy()
        try:
            self.model.geom_group[:] = self.ray_groups
            for angle in self.angles:
                direction = np.array([math.cos(yaw + angle), math.sin(yaw + angle), 0.0])
                distance = mujoco.mj_ray(
                    self.model, data, origin, direction, self.groups, 1, -1, hit
                )
                ranges.append(float(np.clip(
                    self.range_max if distance < 0 else distance,
                    self.range_min, self.range_max,
                )))
        finally:
            # Keep the viewer's original visual/collision geometry selection.
            self.model.geom_group[:] = original_groups
        return (float(position[0]), float(position[1]), yaw), ranges


def load_bridge(root, lock):
    path = root / "simulate_python" / "unitree_sdk2py_bridge.py"
    if not path.is_file():
        raise FileNotFoundError("Unitree SDK bridge not found: " + str(path))
    spec = importlib.util.spec_from_file_location("sea_unitree_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class LockedBridge(module.UnitreeSdk2Bridge):
        def LowCmdHandler(self, msg):
            with lock:
                return super().LowCmdHandler(msg)

        def PublishLowState(self):
            with lock:
                return super().PublishLowState()

        def PublishHighState(self):
            with lock:
                return super().PublishHighState()

    return LockedBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[5] / "unitree_mujoco"
    parser.add_argument("--unitree-root", type=Path, default=default_root)
    parser.add_argument("--scene", type=Path, help="Defaults to Go2 scene_terrain.xml")
    parser.add_argument("--base-body", default="base_link")
    parser.add_argument("--lidar-height", type=float, default=0.05,
                        help="Vertical offset above base origin in meters")
    parser.add_argument("--publish-hz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Wall seconds before exiting; 0 runs until stopped")
    parser.add_argument("--no-dds", action="store_true",
                        help="Publish perception without the motor bridge (diagnostics only)")
    args = parser.parse_args()
    if args.publish_hz <= 0 or args.dt <= 0 or args.duration < 0:
        parser.error("publish-hz/dt must be positive and duration nonnegative")
    root = args.unitree_root.expanduser().resolve()
    scene = (args.scene.expanduser().resolve() if args.scene else
             root / "unitree_robots" / "go2" / "scene_terrain.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sensor = Perception(model, args.base_body, args.lidar_height)
    lock = threading.RLock()

    import rclpy
    from geometry_msgs.msg import Pose2D
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    node = rclpy.create_node("sea_mujoco_perception")
    rays_pub = node.create_publisher(LaserScan, "/rays", 10)
    pose_pub = node.create_publisher(Pose2D, "/pose", 10)
    bridge = None
    try:
        if not args.no_dds:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(1, "lo")
            bridge = load_bridge(root, lock)(model, data)
        node.get_logger().info(
            "Scene: %s; /rays (41 beams) + /pose at %.1f Hz; SDK %s"
            % (scene, args.publish_hz,
               "disabled" if args.no_dds else "domain=1, interface=lo")
        )
        if args.headless:
            context = nullcontext(None)
        else:
            from mujoco import viewer as mujoco_viewer
            context = mujoco_viewer.launch_passive(model, data)
        with context as viewer:
            started = time.monotonic()
            next_publish = 0.0
            next_view = 0.0
            while rclpy.ok() and (viewer is None or viewer.is_running()):
                tick = time.monotonic()
                if args.duration and tick - started >= args.duration:
                    break
                with lock:
                    mujoco.mj_step(model, data)
                    if data.time + 1e-9 >= next_publish:
                        # Refresh derived body transforms after integrating qpos.
                        mujoco.mj_forward(model, data)
                        pose, ranges = sensor.sample(data)
                        msg = LaserScan()
                        msg.header.stamp = node.get_clock().now().to_msg()
                        msg.header.frame_id = "base_link"
                        msg.angle_min = float(sensor.angles[0])
                        msg.angle_max = float(sensor.angles[-1])
                        msg.angle_increment = float(sensor.angles[1] - sensor.angles[0])
                        msg.scan_time = 1.0 / args.publish_hz
                        msg.range_min, msg.range_max = sensor.range_min, sensor.range_max
                        msg.ranges = ranges
                        rays_pub.publish(msg)
                        pose_pub.publish(Pose2D(x=pose[0], y=pose[1], theta=pose[2]))
                        next_publish = data.time + 1.0 / args.publish_hz
                    if viewer is not None and tick >= next_view:
                        viewer.sync()
                        next_view = tick + 0.02
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(max(0.0, args.dt - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if bridge is not None:
            for name in ("lowStateThread", "HighStateThread", "WirelessControllerThread"):
                getattr(bridge, name).Wait(1.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
