import logging

import rclpy
from rclpy.node import Node

from ros_base.manager.base_manager import BaseManager
from ros_base.utils.decorators import profile_latency


class BaseNode:
    def __init__(self, manager: BaseManager = None, *args, **kwargs):
        """Base class for all nodes. Can be used independently or as part of a BaseManager.
        Args:
            manager (BaseManager, optional): The manager that this node belongs to. Defaults to None (standalone ros node).
        """
        self._manager = manager if manager is not None else Node(*args, **kwargs)

    def create_publisher(self, *args, **kwargs):
        return self._manager.create_publisher(*args, **kwargs)

    def create_subscription(self, *args, **kwargs):
        return self._manager.create_subscription(*args, **kwargs)

    def create_timer(self, *args, **kwargs):
        return self._manager.create_timer(*args, **kwargs)

    def destroy_subscription(self, subscription):
        return self._manager.destroy_subscription(subscription)

    def destroy_publisher(self, publisher):
        return self._manager.destroy_publisher(publisher)

    def destroy_node(self):
        return self._manager.destroy_node()

    def get_clock(self):
        return self._manager.get_clock()

    def release_resources(self):
        """Release resources held by the node."""
        pass

    def start_spin_standalone(self):
        """Start the node in standalone mode (without a manager)."""
        try:
            self.logger.info(f"[Node] Starting {self.__class__.__name__} standalone...")
            rclpy.spin(self._manager)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            self.logger.warning("[Node] Shutdown signal received.")
        except Exception as e:
            self.logger.error(f"[Node] Critical Failure: {e}")
            raise e
        finally:
            self.release_resources()
            if rclpy.ok():
                self._manager.destroy_node()
                # Do not rclpy.shutdown() here if it's meant to be used in a shared rclpy init context
                # but for standalone spin it's usually okay.

    def spin_once(self, timeout_sec: float = 0.1):
        """Spin the node once (without a manager) for handshaking."""
        rclpy.spin_once(self._manager, timeout_sec=timeout_sec)

    @property
    def nodes(self):
        """Access sibling nodes via the manager."""
        return getattr(self._manager, "nodes", None)

    @property
    def agents(self):
        """Access agents via the manager."""
        return getattr(self._manager, "agents", None)

    @property
    def logger(self):
        """Returns the appropriate logger (CustomLogger or ROS 2 Logger)."""
        if hasattr(self._manager, "logger"):
            return self._manager.logger
        return self._manager.get_logger()

    def get_logger(self):
        return self.logger

    @property
    def timestamp(self):
        return getattr(self._manager, "timestamp", 0)

    @property
    def node_freq_hz(self):
        return getattr(self._manager, "node_freq_hz", None)
