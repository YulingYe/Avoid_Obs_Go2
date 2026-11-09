from ros_base.configs.camera.base_camera import (
    CameraProfile,
    Extrinsics,
    Intrinsics,
    ROSStreamConfig,
)


def get_zed_mini(compressed: bool = True) -> CameraProfile:
    """Factory builder for ZED Mini Camera Profile."""

    intrinsics = Intrinsics(
        width=1280,
        height=720,
        fx=731.995849609375,
        fy=731.995849609375,
        cx=620.0855102539062,
        cy=362.5731201171875,
        fov=82.0,
    )
    extrinsics = Extrinsics(translation=[0.35, 0.0, 0.0], angles=[0.0, 0.0, 0.0])

    stream = ROSStreamConfig(
        img_topic="/geometry_msgs/image",
        img_compressed=compressed,
        depth_topic="/geometry_msgs/depth",
        depth_compressed=compressed,
    )

    return CameraProfile(
        name="zed_mini" + ("_compressed" if compressed else ""),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        stream=stream,
    )
