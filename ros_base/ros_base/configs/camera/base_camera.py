from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fov: Optional[float] = None
    D: List[float] = field(default_factory=lambda: [0.0] * 5)


@dataclass
class Extrinsics:
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angles: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ROSStreamConfig:
    img_topic: str = "/camera/color/image_raw"
    img_compressed: bool = False

    depth_topic: Optional[str] = None
    depth_compressed: bool = False

    infra1_topic: Optional[str] = None
    infra2_topic: Optional[str] = None
    rgb_info_topic: Optional[str] = None
    infra1_info_topic: Optional[str] = None
    infra2_info_topic: Optional[str] = None


@dataclass
class CameraProfile:
    name: str
    stream: ROSStreamConfig
    intrinsics: Optional[Intrinsics] = None
    extrinsics: Optional[Extrinsics] = None
    fix_extrinsics: bool = True
    fix_intrinsics: bool = True
