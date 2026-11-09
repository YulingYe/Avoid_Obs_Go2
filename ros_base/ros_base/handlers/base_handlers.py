from rclpy.node import Node

from ros_base.manager.base_manager import BaseManager


class BaseHandlers:
    def __init__(self, manager: BaseManager = None, *args, **kwargs):
        self.manager = manager
        # Shortcut to access nodes and agents directly via self.xxx
        self.nodes = manager.nodes if manager else {}
        self.agents = manager.agents if manager else {}

    def handle(self):
        """Main entry point for task logic. Called periodically by the manager."""
        pass

    def reset(self):
        """Optional reset logic for the handlers and its components."""
        pass

    @property
    def logger(self):
        """Standard access to the manager's logger."""
        return self.manager.logger

    @property
    def timestamp(self):
        """Current loop timestamp from the manager."""
        return self.manager.timestamp

    @property
    def node_freq_hz(self):
        """The execution frequency of the main loop."""
        return self.manager.node_freq_hz

    @property
    def state(self):
        """Current state of the manager's FSM."""
        return self.manager.state

    @state.setter
    def state(self, value):
        """Allow handlers to request a state switch in the manager."""
        self.manager.state = value
