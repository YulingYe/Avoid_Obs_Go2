import os
import queue
import threading
from datetime import datetime
from typing import Union

import cv2
import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from ros_base.configs.camera import CameraProfile, get_realsense_d435i, get_zed_mini
from ros_base.nodes.base_node import BaseNode

INVALID_MEAS = -1.0


class CamSubNode(BaseNode):
    def __init__(
        self,
        config: Union[str, dict, CameraProfile] = "realsense_d435i_align",
        record: bool = False,
        record_fps: int = 20,
        vis_rgb: bool = False,
        vis_depth: bool = False,
        gamma: float = 2.0,
        use_clahe: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Resolve config
        if isinstance(config, str):
            if config == "realsense_d435i":
                self.config = get_realsense_d435i(align_depth=False, compressed=False)
            elif config == "realsense_d435i_compressed":
                self.config = get_realsense_d435i(align_depth=False, compressed=True)
            elif config == "realsense_d435i_align":
                self.config = get_realsense_d435i(align_depth=True, compressed=False)
            elif config == "realsense_d435i_align_compressed":
                self.config = get_realsense_d435i(align_depth=True, compressed=True)
            elif config == "realsense_d435i_infra":
                self.config = get_realsense_d435i(align_depth=False, compressed=False, infra=True)
            elif config == "zed_mini":
                self.config = get_zed_mini(compressed=True)
            else:
                error_msg = f"Unsupported camera type string: {config}."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        elif isinstance(config, CameraProfile):
            self.config = config
        else:
            raise ValueError("config must be a string or a CameraProfile object.")

        self.record = record
        self.record_fps = record_fps
        self.vis_rgb = vis_rgb
        self.vis_depth = vis_depth
        self.gamma = gamma
        self.use_clahe = use_clahe
        self.min_record_interval = 1.0 / self.record_fps if self.record_fps > 0 else 0.0
        self.last_record_time = 0.0
        self.record_thread = None
        self.video_writer = None
        self.timestamp_file = None

        if self.record:
            # Recording queue and thread
            self.record_queue = queue.Queue(maxsize=100)
            self.recording_active = False

            self.recording_active = True
            self.record_thread = threading.Thread(target=self._record_worker)
            self.record_thread.start()

        # Camera specific attributes
        self.infra1 = None
        self.infra2 = None
        self.infra_fx = None
        self.infra_baseline = None
        self.rgb_info = None
        self.infra1_info = None
        self.T_rgb_to_infra = None
        self.img = None
        self.img_timestamp = None
        self.img_shape = None
        self.depth = None
        self.depth_timestamp = None
        self.depth_shape = None

        self._setup_camera_subscriptions()

        # Periodic visualization timer (e.g., 20Hz)
        # ONLY start if explicit visualization is requested to avoid threading conflicts
        self.viz_timer = None
        if self.vis_rgb:
            self.viz_timer = self.create_timer(0.05, self.visualize_img)

    def _setup_camera_subscriptions(self):
        self.logger.info(f"Using camera config: {self.config.name}")

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # RGB Image Subscription
        if self.config.stream.img_compressed:
            self.img_msg_type = CompressedImage
        else:
            self.img_msg_type = Image

        self.img_sub = self.create_subscription(
            self.img_msg_type, self.config.stream.img_topic, self._img_callback, self.qos_profile
        )
        self.logger.info(
            f"Subscribed to RGB topic: {self.config.stream.img_topic} (Compressed: {self.config.stream.img_compressed})"
        )

        # Depth Image Subscription
        if self.config.stream.depth_topic:
            if self.config.stream.depth_compressed:
                self.depth_msg_type = CompressedImage
            else:
                self.depth_msg_type = Image

            self.depth_sub = self.create_subscription(
                self.depth_msg_type, self.config.stream.depth_topic, self._depth_callback, self.qos_profile
            )
            self.logger.info(
                f"Subscribed to Depth topic: {self.config.stream.depth_topic} (Compressed:"
                f" {self.config.stream.depth_compressed})"
            )

        # Infra Subscriptions
        if self.config.stream.infra1_topic:
            self._setup_infra_subscriptions()

        # RGB Info Subscription
        if self.config.stream.rgb_info_topic:
            self.rgb_info_sub = self.create_subscription(
                CameraInfo, self.config.stream.rgb_info_topic, self._rgb_info_callback, self.qos_profile
            )

    def _setup_infra_subscriptions(self):
        if self.config.stream.infra1_topic:
            self.infra1_sub = self.create_subscription(
                Image, self.config.stream.infra1_topic, self._infra1_callback, self.qos_profile
            )
        if self.config.stream.infra2_topic:
            self.infra2_sub = self.create_subscription(
                Image, self.config.stream.infra2_topic, self._infra2_callback, self.qos_profile
            )
        if self.config.stream.infra1_info_topic:
            self.infra1_info_sub = self.create_subscription(
                CameraInfo, self.config.stream.infra1_info_topic, self._infra1_info_callback, self.qos_profile
            )
        if self.config.stream.infra2_info_topic:
            self.infra2_info_sub = self.create_subscription(
                CameraInfo, self.config.stream.infra2_info_topic, self._infra2_info_callback, self.qos_profile
            )

    def _record_worker(self):
        while self.recording_active:
            try:
                item = self.record_queue.get(timeout=1.0)

                # Check for poison pill (using "is" to avoid numpy array truth value errors)
                if item[0] is None and item[1] is None:
                    break

                img, timestamp = item

                if self.video_writer is None:
                    self._init_video_writer(img.shape)

                self.video_writer.write(img)
                self.timestamp_file.write(f"{timestamp.sec}.{timestamp.nanosec:09d}\n")
                self.timestamp_file.flush()
                self.record_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in recording thread: {e}")

    def _init_video_writer(self, shape):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = datetime.now().strftime("track_session_%Y%m%d_%H%M")
        save_dir = os.path.expanduser(f"~/Data/rosbags/{session_folder}")
        os.makedirs(save_dir, exist_ok=True)

        video_path = os.path.join(save_dir, f"cam_sub_img_{timestamp}.mp4")
        ts_path = os.path.join(save_dir, f"cam_sub_ts_{timestamp}.txt")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = float(self.record_fps)
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (shape[1], shape[0]))
        self.timestamp_file = open(ts_path, "w")
        self.logger.info(f"Recording video to: {video_path}")
        self.logger.info(f"Recording timestamps to: {ts_path}")

    def release_resources(self):
        self.recording_active = False

        # Unblock the thread if it's waiting on empty queue
        if hasattr(self, "record_queue"):
            self.record_queue.put_nowait((None, None))

        if self.record_thread:
            self.record_thread.join(timeout=2.0)

        if self.video_writer:
            self.video_writer.release()
            self.logger.info("Video writer released.")
        if self.timestamp_file:
            self.timestamp_file.close()
            self.logger.info("Timestamp file closed.")

    def _img_callback(self, msg):
        if self.config.stream.img_compressed:
            self.img_jpg_bytes = msg.data
            img_raw = cv2.imdecode(np.frombuffer(self.img_jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            img_raw = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, -1))
            # Fix color space if incoming is RGB (RealSense default is rgb8, OpenCV expects bgr8)
            if hasattr(msg, "encoding") and "rgb" in msg.encoding:
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)

        # Apply preprocessing (Gamma, CLAHE, etc.)
        if img_raw is not None:
            self.img = self._preprocess_image(img_raw)
        else:
            self.img = None

        if self.record and self.img is not None:
            current_msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if current_msg_time - self.last_record_time >= self.min_record_interval:
                if not self.record_queue.full():
                    self.record_queue.put((self.img, msg.header.stamp))
                    self.last_record_time = current_msg_time
                else:
                    self.logger.warning("Recording queue full, dropping frame")

        self.img_timestamp = msg.header.stamp
        self.img_shape = self.img.shape

    def _depth_callback(self, msg):
        if self.config.stream.depth_compressed:
            self.depth_png_bytes = msg.data
            self.depth = cv2.imdecode(np.frombuffer(self.depth_png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            self.depth = np.frombuffer(msg.data, np.uint16).reshape((msg.height, msg.width))

        self.depth_shape = self.depth.shape
        self.depth_timestamp = msg.header.stamp
        if self.vis_depth:
            self.visualize_depth()

    def _infra1_callback(self, msg):
        self.infra1 = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width))

    def _infra2_callback(self, msg):
        self.infra2 = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width))

    def _infra1_info_callback(self, msg):
        self.infra1_info = msg
        self.infra_fx = msg.k[0]
        self.infra_baseline = -msg.p[3] / msg.p[0]

    def _infra2_info_callback(self, msg):
        pass

    def _rgb_info_callback(self, msg):
        self.rgb_info = msg

    def _preprocess_image(self, img):
        """Perform image enhancement to reduce shadow impact."""
        # 1. Gamma Correction
        if self.gamma != 1.0:
            invGamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)

        # 2. CLAHE (Local Contrast Enhancement)
        if self.use_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # ClipLimit 2-3 is usually good for shadows
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        return img

    def visualize_depth(self):
        if self.depth is None:
            return
        depth_vis = (self.depth / np.max(self.depth) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow("Depth Image", depth_colormap)
        cv2.waitKey(10)

    def visualize_img(self):
        """Draw active overlays on the image and show it."""
        if self.img is not None:
            cv2.imshow("RGB Image", self.img)
            cv2.waitKey(1)
        else:
            # If nothing to visualize, close window if it exists
            try:
                if cv2.getWindowProperty("RGB Image", cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow("RGB Image")
                    cv2.waitKey(1)
            except Exception:
                pass

    def reset(self):
        self.img = None
        self.depth = None
        self.infra1 = None
        self.infra2 = None


if __name__ == "__main__":
    import rclpy

    from ros_base.utils.args_debug import add_debug_mode

    parser = add_debug_mode(listen_port=8888)
    parser.add_argument(
        "--camera", type=str, default="realsense_d435i_align", help="Camera type: realsense_d435i or zed_mini."
    )
    parser.add_argument("--record", action="store_true", help="Record images to rosbag.")
    parser.add_argument("--record_fps", type=int, default=20, help="Recording FPS.")
    parser.add_argument("--vis_rgb", action="store_true", help="Whether visualize RGB image.")
    parser.add_argument("--vis_depth", action="store_true", help="Whether visualize Depth image.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value (>1.0 to brighten).")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE local contrast enhancement.")
    args = parser.parse_args()

    rclpy.init()
    cam_node = CamSubNode(
        node_name="cam_sub_node",
        config=args.camera,
        record=args.record,
        record_fps=args.record_fps,
        vis_rgb=args.vis_rgb,
        vis_depth=args.vis_depth,
        gamma=args.gamma,
        use_clahe=args.clahe,
    )
    try:
        cam_node.start_spin_standalone()
    except KeyboardInterrupt:
        pass
    finally:
        cam_node.release_resources()
        if rclpy.ok():
            cam_node.destroy_node()
            rclpy.shutdown()
