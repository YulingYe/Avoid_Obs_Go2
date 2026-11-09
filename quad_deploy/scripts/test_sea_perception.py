"""Headless geometric regression checks; needs numpy and mujoco, no ROS/DDS."""
import math

import mujoco
import numpy as np

from quad_deploy.scripts.sea.mujoco_sim import Perception


def main():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody>
      <geom type="box" pos="2 0 0.5" size="0.1 0.5 0.5"/>
      <body name="base_link" pos="0 0 0.5">
        <freejoint/><geom type="sphere" size="0.2"/>
        <body pos="0.4 0 0"><geom type="sphere" size="0.1"/></body>
      </body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    original_groups = model.geom_group.copy()
    sensor = Perception(model, lidar_height=0.0)
    mujoco.mj_forward(model, data)
    pose, rays = sensor.sample(data)
    np.testing.assert_array_equal(model.geom_group, original_groups)
    assert len(rays) == 41
    np.testing.assert_allclose(pose, [0, 0, 0], atol=1e-9)
    # Both the base geometry and child sphere must be excluded.
    np.testing.assert_allclose(rays[20], 1.9, atol=1e-8)
    assert rays[0] == rays[-1] == 3.0
    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    pose, rays = sensor.sample(data)
    np.testing.assert_allclose(pose[0], 1.0)
    np.testing.assert_allclose(rays[20], 0.9, atol=1e-8)
    data.qpos[3:7] = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    mujoco.mj_forward(model, data)
    pose, rays = sensor.sample(data)
    np.testing.assert_allclose(pose[2], math.pi / 2, atol=1e-8)
    assert rays[20] == 3.0  # forward rotates with the robot
    np.testing.assert_allclose(rays[5], 0.9, atol=1e-8)  # -90 degrees sees wall
    print("PASS: distance, no-return, subtree exclusion, translation, yaw and beam order")


if __name__ == "__main__":
    main()
