import functools
import time
from typing import Optional


def profile_latency(
    cycle_threshold_ms: float = 50.0,
    process_threshold_ms: float = 20.0,
    log_interval_s: float = 5.0,
    debug: bool = True,
    check_capture_latency: bool = False,
):
    """
    Decorator to monitor cycle latency (time between calls) and processing latency (execution time).

    Args:
        cycle_threshold_ms (float): Threshold for time between consecutive calls in milliseconds.
        process_threshold_ms (float): Threshold for function execution time in milliseconds.
        log_interval_s (float): Interval in seconds to log average latency stats. Set to 0 to disable.
        debug (bool): If True, log warnings when thresholds are exceeded.
        check_capture_latency (bool): If True, check latency between message capture time and receive time.
                                      Requires the first argument of the decorated function to be a ROS message
                                      with a header.stamp field.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Try to get ROS clock, otherwise fall back to system time
            if hasattr(self, "get_clock"):
                try:
                    now = self.get_clock().now()
                    current_time = now.nanoseconds * 1e-9
                except Exception:
                    current_time = time.time()
            else:
                current_time = time.time()

            func_name = func.__name__

            # --- Capture Latency Check ---
            if check_capture_latency and len(args) > 0:
                msg = args[0]
                if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                    capture_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    latency = current_time - capture_time
                    if hasattr(self, "logger"):
                        self.logger.info(
                            f"[{func_name} Check] Recv Time: {current_time:.3f}, Capture Time: {capture_time:.3f}, "
                            f"Latency: {latency*1000:.1f}ms",
                            # throttle_duration_sec=1.0,  # Log every 1 second
                        )

            # Initialize storage for latency stats if not present
            if not hasattr(self, "_latency_stats"):
                self._latency_stats = {}

            # Keys for storage
            last_call_key = f"{func_name}_last_call"
            last_log_key = f"{func_name}_last_log"
            cycle_sum_key = f"{func_name}_cycle_sum"
            cycle_max_key = f"{func_name}_cycle_max"
            process_sum_key = f"{func_name}_process_sum"
            process_max_key = f"{func_name}_process_max"
            count_key = f"{func_name}_count"

            # --- Cycle Latency Check ---
            last_call_time = self._latency_stats.get(last_call_key)
            dt_ms = 0.0

            if last_call_time is not None:
                dt_ms = (current_time - last_call_time) * 1000.0
                if debug and dt_ms > cycle_threshold_ms:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"[{func_name}] Cycle latency {dt_ms:.2f} ms > threshold {cycle_threshold_ms} ms"
                        )

            # Update last call time
            self._latency_stats[last_call_key] = current_time

            # --- Processing Latency Check ---
            t0 = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
            finally:
                t1 = time.perf_counter()
                process_ms = (t1 - t0) * 1000.0

                if debug and process_ms > process_threshold_ms:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"[{func_name}] Processing time {process_ms:.2f} ms > threshold {process_threshold_ms} ms"
                        )

                # --- Periodic Logging ---
                if log_interval_s > 0:
                    # Initialize stats if needed
                    if self._latency_stats.get(last_log_key) is None:
                        self._latency_stats[last_log_key] = current_time
                        self._latency_stats[cycle_sum_key] = 0.0
                        self._latency_stats[cycle_max_key] = 0.0
                        self._latency_stats[process_sum_key] = 0.0
                        self._latency_stats[process_max_key] = 0.0
                        self._latency_stats[count_key] = 0

                    # Accumulate (skip first cycle for cycle_sum as dt_ms is 0)
                    if last_call_time is not None:
                        self._latency_stats[cycle_sum_key] += dt_ms
                        if dt_ms > self._latency_stats[cycle_max_key]:
                            self._latency_stats[cycle_max_key] = dt_ms

                    self._latency_stats[process_sum_key] += process_ms
                    if process_ms > self._latency_stats[process_max_key]:
                        self._latency_stats[process_max_key] = process_ms

                    self._latency_stats[count_key] += 1

                    # Check interval
                    if current_time - self._latency_stats[last_log_key] > log_interval_s:
                        count = self._latency_stats[count_key]
                        if count > 0:
                            avg_cycle = self._latency_stats[cycle_sum_key] / count
                            max_cycle = self._latency_stats[cycle_max_key]
                            avg_process = self._latency_stats[process_sum_key] / count
                            max_process = self._latency_stats[process_max_key]

                            if hasattr(self, "logger"):
                                self.logger.info(
                                    f"[{func_name}] Stats ({log_interval_s}s): "
                                    f"Cycle Avg/Max {avg_cycle:.2f}/{max_cycle:.2f} ms, "
                                    f"Process Avg/Max {avg_process:.2f}/{max_process:.2f} ms"
                                )

                        # Reset
                        self._latency_stats[last_log_key] = current_time
                        self._latency_stats[cycle_sum_key] = 0.0
                        self._latency_stats[cycle_max_key] = 0.0
                        self._latency_stats[process_sum_key] = 0.0
                        self._latency_stats[process_max_key] = 0.0
                        self._latency_stats[count_key] = 0

            return result

        return wrapper

    return decorator
