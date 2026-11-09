from typing import Optional

from ros_base.configs.camera.base_camera import (
    CameraProfile,
    Extrinsics,
    Intrinsics,
    ROSStreamConfig,
)


def get_realsense_d435i(
    resolution: str = "320x180", align_depth: bool = True, compressed: bool = False, infra: bool = False
) -> CameraProfile:
    """Factory builder for Realsense D435i Camera Profile."""

    if resolution == "320x180":
        intrinsics = Intrinsics(
            width=320,
            height=180,
            fx=227.3841552734375,
            fy=227.24505615234375,
            cx=162.8849639892578,
            cy=92.34309387207031,
        )
    elif resolution == "320x240":
        intrinsics = Intrinsics(
            width=320, height=240, fx=303.1788635253906, fy=302.993408203125, cx=163.8466033935547, cy=123.1241226196289
        )
    elif resolution == "640x360":
        intrinsics = Intrinsics(
            width=640,
            height=360,
            fx=454.768310546875,
            fy=454.4901123046875,
            cx=325.7699279785156,
            cy=184.68618774414062,
            fov=70.26,
        )
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    extrinsics = Extrinsics(translation=[0.305, 0.017, 0.128], angles=[0.0, 30.0, 0.0])

    depth_topic = "/camera/aligned_depth_to_color/image_raw" if align_depth else "/camera/depth/image_rect_raw"
    img_topic = "/camera/color/image_raw/compressed" if compressed else "/camera/color/image_raw"

    stream = ROSStreamConfig(
        img_topic=img_topic,
        img_compressed=compressed,
        depth_topic=depth_topic,
        depth_compressed=compressed,
        rgb_info_topic="/camera/color/camera_info",
    )

    if infra:
        stream.infra1_topic = "/camera/infra1/image_rect_raw"
        stream.infra2_topic = "/camera/infra2/image_rect_raw"
        stream.infra1_info_topic = "/camera/infra1/camera_info"
        stream.infra2_info_topic = "/camera/infra2/camera_info"

    return CameraProfile(
        name=f"realsense_d435i_{resolution}"
        + ("_align" if align_depth else "")
        + ("_infra" if infra else "")
        + ("_compressed" if compressed else ""),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        stream=stream,
    )
