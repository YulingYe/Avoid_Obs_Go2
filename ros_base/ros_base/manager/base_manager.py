import multiprocessing
import os
import signal
import threading
import time
from typing import Dict, List, Optional, Type

import rclpy
from rclpy.node import Node

from ros_base.utils.logger import CustomLogger

# Default global logger for setup phase
logger = CustomLogger("BaseManager")


def run_node_process(node_class: Type, node_name: str, args: tuple, kwargs: dict):
    """Helper function to run a node in a separate process."""
    try:
        # Each process needs its own ROS context
        if not rclpy.ok():
            rclpy.init(args=None)
        node = node_class(node_name=node_name, *args, **kwargs)
        node.start_spin_standalone()
    except Exception as e:
        logger.error(f"Process {node_name} failed: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def register_multiprocess_nodes(mp_nodes_dict: dict = None, cmds_dict: dict = None, *args, **kwargs):
    """Register and start nodes in separate processes."""
    processes = []

    # 1. Start External Commands (e.g. bash scripts)
    if cmds_dict:
        for cmd_name, cmd in cmds_dict.items():
            process = multiprocessing.Process(target=os.system, args=(cmd,), name=f"cmd_{cmd_name}", daemon=True)
            process.start()
            logger.info(f"Started command({cmd_name}): {cmd}")
            processes.append(process)
            time.sleep(0.5)

    # 2. Start Sub-Nodes (as separate Python processes)
    if mp_nodes_dict:
        for node_name, node_class in mp_nodes_dict.items():
            process = multiprocessing.Process(
                target=run_node_process,
                args=(node_class, node_name, args, kwargs),
                name=f"node_{node_name}",
                daemon=True,
            )
            process.start()
            logger.info(f"Started {node_name} node in a new process.")
            processes.append(process)
            time.sleep(0.5)

    return processes


def shutdown_multiprocess_nodes(processes):
    """Cleanly shutdown all child processes."""
    if not processes:
        return

    logger.info(f"Shutting down {len(processes)} child processes...")

    # First attempt: SIGTERM (request exit)
    for process in processes:
        if process.is_alive():
            process.terminate()

    # Wait for processes to exit
    start_wait = time.time()
    while time.time() - start_wait < 3.0:  # Wait up to 3 seconds
        if all(not p.is_alive() for p in processes):
            break
        time.sleep(0.1)

    # Second attempt: SIGKILL (force exit if still alive)
    for process in processes:
        if process.is_alive():
            logger.warning(f"Process {process.name} still alive, killing it...")
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.join()

    logger.info("All child processes cleaned up.")


class BaseManager(Node):
    def __init__(
        self,
        node_name: str = "BaseManager",
        nodes_dict: dict = {},
        agents_dict: dict = {},
        handlers_class: type = None,
        node_freq_hz: int = 200,
        start_state: str = None,
        custom_logger: type = None,
        log_freq: bool = False,
        *args,
        **kwargs,
    ):
        """
        Base Manager to orchestrate nodes, agents, and state handlers.
        """
        super().__init__(node_name)

        # Configuration
        self.node_freq_hz = node_freq_hz
        self.log_freq = log_freq

        # State
        self.state = start_state
        self.timestamp = 0
        self.start_time = time.monotonic()

        # Storage
        self.nodes: Dict[str, any] = {}
        self.agents: Dict[str, any] = {}
        self.handlers: any = None
        self._handshake_rules: Dict[str, callable] = {}

        # Logger setup
        if custom_logger is not None:
            self.logger = custom_logger(node_name=node_name, *args, **kwargs)
        else:
            self.logger = self.get_logger()

        # Initialization
        self._register_nodes(nodes_dict, *args, **kwargs)
        self._register_agents(agents_dict, *args, **kwargs)
        self._register_handlers(handlers_class, *args, **kwargs)

    def _register_agents(self, agents_dict, *args, **kwargs):
        """Register agents from the agents_dict. Deliver the manager instance to each agent."""
        for agent_name, agent_class in agents_dict.items():
            try:
                self.agents[agent_name] = agent_class(manager=self, *args, **kwargs)
                self.logger.info(f"Successfully registered {agent_name} agent")
            except Exception as e:
                self.logger.error(f"Failed to register agent {agent_name}: {e}")

    def _register_nodes(self, nodes_dict, *args, **kwargs):
        """Register nodes from the nodes_dict. Deliver the manager instance to each node."""
        for node_name, node_class in nodes_dict.items():
            try:
                self.nodes[node_name] = node_class(manager=self, *args, **kwargs)
                self.logger.info(f"Successfully registered {node_name} node")
            except Exception as e:
                self.logger.error(f"Failed to register node {node_name}: {e}")

    def _register_handlers(self, handlers_class=None, *args, **kwargs):
        """Register handlers from the handlers_class. Deliver the manager instance to the handlers."""
        if handlers_class is not None:
            try:
                self.handlers = handlers_class(manager=self, *args, **kwargs)
                self.logger.info(f"Successfully registered {handlers_class.__name__} handlers")
            except Exception as e:
                self.logger.error(f"Failed to register handlers {handlers_class.__name__}: {e}")

    def add_handshake_rule(self, name: str, condition: callable):
        """Register a new handshake condition."""
        self._handshake_rules[name] = condition

    def handshake(self) -> bool:
        """Standard rules-based handshake. Override if complex logic is needed."""
        if not self._handshake_rules:
            return True

        ready_count = 0
        for name, condition in self._handshake_rules.items():
            self.logger.log_once(f"Handshake check: {name}...")
            if condition():
                ready_count += 1
                self.logger.info(f"Handshake verified: {name}", once=True)

        return ready_count == len(self._handshake_rules)

    def release_resources(self):
        """Clean up local resources."""
        pass

    def start_main_loop_timer(self, processes: Optional[List] = None, *args, **kwargs):
        """Entry point for the management loop."""
        # Wait for handshake
        self.logger.info("Entering handshake phase...")
        while rclpy.ok():
            if self.handshake():
                self.logger.info("Handshake successful.")
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        timer_period = 1.0 / self.node_freq_hz
        self.timer = self.create_timer(timer_period, self.main_loop)
        self.start_time = time.monotonic()

        try:
            self.logger.info(f"Manager {self.get_name()} spinning at {self.node_freq_hz}Hz...")
            rclpy.spin(self)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            self.logger.warning("Shutdown received, exiting loop...")
        except Exception as e:
            self.logger.error(f"Loop Exception: {e}")
            raise e
        finally:
            if processes:
                shutdown_multiprocess_nodes(processes)
            self.release_resources()
            self.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    def main_loop(self):
        """Execution sequence called by the timer."""
        # 1. State Machine Handling (Internal logic)
        if hasattr(self, "get_state_switch") and callable(self.get_state_switch):
            new_state = self.get_state_switch()
            if new_state is not None and new_state != self.state:
                if hasattr(self.logger, "important"):
                    self.logger.important(f"State Switch: {self.state} -> {new_state}")
                else:
                    self.logger.info(f"State Switch: {self.state} -> {new_state}")
                self.state = new_state

            if hasattr(self, "state_handle") and callable(self.state_handle):
                self.state_handle(new_state)

        # 2. Handler logic (Task flow)
        if self.handlers and hasattr(self.handlers, "handle"):
            self.handlers.handle()

        # 3. Bookkeeping
        self.timestamp += 1
        self._log_main_freq()

    def _log_main_freq(self):
        if self.log_freq and self.timestamp % 1000 == 0:
            elapsed = time.monotonic() - self.start_time
            freq = self.timestamp / elapsed
            self.logger.info(f"Current Frequency: {freq:.2f} Hz")
