from abc import ABC, abstractmethod

from ros_base.manager.base_manager import BaseManager
from ros_base.utils.logger import CustomLogger


class BaseAgent(ABC):
    """
    Base class for generic agents in the ROS 2 hierarchical manager system.
    This class defines the interface that all agents must implement.
    """

    def __init__(self, manager: BaseManager = None, *args, **kwargs):
        self._manager = manager
        self._logger = CustomLogger(self.__class__.__name__)

    def handle(self):
        """Run the agent's main loop."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the agent. This is a placeholder for any reset logic if needed."""
        pass

    def get_clock(self):
        if self._manager is None:
            raise ValueError("Manager is not set for this agent.")
        return self._manager.get_clock()

    @property
    def logger(self):
        if self._manager is None:
            return self._logger
        return self._manager.logger

    @property
    def nodes(self):
        if self._manager is None:
            raise ValueError("Manager is not set for this agent.")
        return self._manager.nodes

    @property
    def agents(self):
        if self._manager is None:
            raise ValueError("Manager is not set for this agent.")
        return self._manager.agents

    @property
    def timestamp(self):
        if self._manager is None:
            raise ValueError("Manager is not set for this agent.")
        return self._manager.timestamp

    @property
    def state(self):
        if self._manager is None:
            raise ValueError("Manager is not set for this agent.")
        return self._manager.state

    @property
    def node_freq_hz(self):
        if self._manager is None:
            raise ValueError("Manager is not set for this agent.")
        return self._manager.node_freq_hz
