import numpy as np

from ros_base.utils.logger import CustomLogger

logger = CustomLogger(__name__)

INVALID_DEPTH = 0.0  # used when depth is invalid
DEPTH_MIN = 0.1  # in m, used when depth is NaN
DEPTH_MAX = 10.0  # in m, used when depth is Inf


def get_depth_at_pixel_depth(current_depth=None, u=None, v=None, u_norm=None, v_norm=None, unit="mm"):
    if current_depth is None:
        logger.log_once(f"No depth data available. Return {INVALID_DEPTH}", level="ERROR")
        return INVALID_DEPTH

    if unit == "m":
        conversion_factor = 1.0
    elif unit == "mm":
        conversion_factor = 0.001
    else:
        logger.log_once(f"Unknown unit '{unit}'. Defaulting to mm.", level="WARNING")
        conversion_factor = 0.001

    depth_width = current_depth.shape[1]
    depth_height = current_depth.shape[0]

    if u is None or v is None:
        if u_norm is None or v_norm is None:
            logger.log_once(f"No pixel coordinates provided. Return {INVALID_DEPTH}", level="ERROR")
            return INVALID_DEPTH
        u = int(u_norm * depth_width)
        v = int(v_norm * depth_height)

    u = max(0, min(u, depth_width - 1))
    v = max(0, min(v, depth_height - 1))

    try:
        depth = float(current_depth[v, u])
        if np.isfinite(depth) and depth > INVALID_DEPTH:
            return depth * conversion_factor
        if np.isnan(depth):
            return DEPTH_MIN * conversion_factor
        if np.isinf(depth):
            return DEPTH_MAX * conversion_factor

        search_radius = 2
        u_min = max(0, u - search_radius)
        u_max = min(depth_width, u + search_radius + 1)
        v_min = max(0, v - search_radius)
        v_max = min(depth_height, v + search_radius + 1)
        neighborhood = current_depth[v_min:v_max, u_min:u_max]
        valid_mask = (neighborhood > INVALID_DEPTH) & np.isfinite(neighborhood)
        valid_values = neighborhood[valid_mask]

        if valid_values.size > 0:
            median_depth = np.median(valid_values)
            return float(median_depth) * conversion_factor
        else:
            return INVALID_DEPTH
    except Exception as e:
        logger.error(f"Failed to get depth at pixel ({u}, {v}): {e}")
        return INVALID_DEPTH


def get_depth_at_pixel_point_cloud(current_point_cloud=None, u=None, v=None, u_norm=None, v_norm=None):
    if current_point_cloud is None:
        logger.log_once(f"No point cloud data available. Return {INVALID_DEPTH}", level="WARNING")
        return INVALID_DEPTH

    depth_width = current_point_cloud.shape[1]
    depth_height = current_point_cloud.shape[0]

    if u is None or v is None:
        if u_norm is None or v_norm is None:
            logger.log_once(f"No pixel coordinates provided. Return {INVALID_DEPTH}", level="WARNING")
            return INVALID_DEPTH
        u = int(u_norm * depth_width)
        v = int(v_norm * depth_height)

    try:
        depth = float(current_point_cloud[v, u])
        if np.isfinite(depth) and depth > INVALID_DEPTH:
            return depth
        elif np.isnan(depth):
            return DEPTH_MIN
        elif np.isinf(depth):
            return DEPTH_MAX
        else:
            logger.warning(f"Depth invalid at pixel ({u}, {v}): {depth}")
            return INVALID_DEPTH
    except Exception as e:
        logger.error(f"Failed to get depth at pixel ({u}, {v}): {e}")
        return INVALID_DEPTH
