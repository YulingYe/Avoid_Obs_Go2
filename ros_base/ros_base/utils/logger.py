import sys
import time
from typing import Optional, Set

from loguru import logger


class CustomLogger:
    """A custom logger class based on Loguru with ROS 2 compatible interface."""

    def __init__(
        self,
        name: Optional[str] = None,
        log_file: Optional[str] = None,
        level: str = "INFO",
        rotation: str = "10 MB",
        format: str = "<green>{time:YY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <white>{message}</white>",
        *args,
        **kwargs,
    ):
        """
        Initialize the custom logger.

        Args:
            name: Logger name. If None, uses 'node_name' from kwargs.
            log_file: Optional path to a log file.
            level: Logging level (e.g., 'INFO', 'DEBUG').
            rotation: Rotation policy for the log file.
            format: Format string for the console output.
        """
        # Resolve name from node_name if not provided directly (BaseManager pattern)
        if name is None:
            name = kwargs.get("node_name", "Logger")

        self.throttle_times = {}
        self.printed_messages: Set[str] = set()

        # Clear existing sinks to avoid duplicate output
        logger.remove()

        # Add console sink
        logger.add(
            sys.stderr,
            level=level,
            format=format,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # Add file sink if requested
        if log_file:
            # Strip tags for plain text file output
            clean_format = (
                format.replace("<level>", "")
                .replace("</level>", "")
                .replace("<green>", "")
                .replace("</green>", "")
                .replace("<white>", "")
                .replace("</white>", "")
                .replace("<cyan>", "")
                .replace("</cyan>", "")
                .replace("<yellow>", "")
                .replace("</yellow>", "")
                .replace("<bold>", "")
                .replace("</bold>", "")
            )
            logger.add(
                log_file,
                rotation=rotation,
                level=level,
                format=clean_format,
                colorize=False,
            )

        self._logger = logger.bind(name=name)

    def _should_log(self, message: str, once: bool, throttle: float) -> bool:
        """Determines if a message should be logged based on once/throttle constraints."""
        if once and message in self.printed_messages:
            return False

        if throttle > 0:
            current_time = time.time()
            last_log_time = self.throttle_times.get(message, 0)
            if current_time - last_log_time < throttle:
                return False
            self.throttle_times[message] = current_time

        if once:
            self.printed_messages.add(message)

        return True

    # --- ROS 2 Standard Interface ---

    def debug(self, msg: str, once: bool = False, throttle: float = 0.0):
        if self._should_log(msg, once, throttle):
            self._logger.debug(msg)

    def info(self, msg: str, once: bool = False, throttle: float = 0.0):
        if self._should_log(msg, once, throttle):
            self._logger.info(msg)

    def warning(self, msg: str, once: bool = False, throttle: float = 0.0):
        if self._should_log(msg, once, throttle):
            self._logger.warning(msg)

    def warn(self, msg: str, once: bool = False, throttle: float = 0.0):
        """ROS 2 alias for warning."""
        self.warning(msg, once=once, throttle=throttle)

    def error(self, msg: str, once: bool = False, throttle: float = 0.0):
        if self._should_log(msg, once, throttle):
            self._logger.error(msg)

    def fatal(self, msg: str, once: bool = False, throttle: float = 0.0):
        """ROS 2 alias for critical."""
        if self._should_log(msg, once, throttle):
            self._logger.critical(msg)

    def critical(self, msg: str, once: bool = False, throttle: float = 0.0):
        """Alias for fatal."""
        self.fatal(msg, once=once, throttle=throttle)

    # --- Feature Methods (Compatibility) ---

    def log_once(self, message: str, level: str = "INFO"):
        """Legay method for once-only logging."""
        self._log_at_level(level, message, once=True)

    def log_throttle(self, message: str, seconds: float, level: str = "INFO"):
        """Legacy method for throttled logging."""
        self._log_at_level(level, message, throttle=seconds)

    def important(self, message: str, level: str = "INFO"):
        """Logs an important message highlighted at the level's color with yellow text."""
        # Use loguru colors via opt(colors=True)
        log_func = getattr(self._logger.opt(colors=True), level.lower(), self._logger.info)
        log_func(f"<yellow>{message}</yellow>")

    def _log_at_level(self, level_name: str, message: str, once=False, throttle=0.0):
        """Internal helper to call the right logging function by string name."""
        func = getattr(self, level_name.lower(), self.info)
        func(message, once=once, throttle=throttle)

    def reset(self):
        """Resets the printed_messages set and throttle_times dict."""
        self.printed_messages.clear()
        self.throttle_times.clear()


# Example usage
if __name__ == "__main__":
    # Initialize logger with terminal and file output
    custom_logger = CustomLogger(name="my_app", log_file="app.log")

    # Test log_once
    for _ in range(3):
        custom_logger.log_once("This will only print once", level="INFO")
        time.sleep(1)

    # Test log_throttle
    for _ in range(6):
        custom_logger.log_throttle("This will print every 2 seconds", seconds=2, level="WARNING")
        time.sleep(1)

    # Test important
    custom_logger.important("This is an important message", level="INFO")

    # Test regular log
    custom_logger.debug("This is debug")
    custom_logger.info("This is info")
    custom_logger.warning("This is warning")
    custom_logger.error("This is error")
    custom_logger.critical("This is critical")
