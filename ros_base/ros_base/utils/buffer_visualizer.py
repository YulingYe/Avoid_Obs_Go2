import matplotlib.pyplot as plt
import numpy as np


class BufferVisualizer:
    def __init__(
        self,
        *buffers,
        labels=None,
        title="Data Comparison",
        subplot_titles=None,
        data_extractor=None,
        time_extractor=None,
        use_index_as_time=False,
        **kwargs,
    ):
        """
        Real-time visualizer for comparing multiple data buffers.

        Args:
            *buffers: Variable number of iterable buffers (e.g. deque, list).
            labels: List of legend labels corresponding to buffers.
            title: Figure title.
            subplot_titles: List of titles for subplots. Determines number of plots.
                            Defaults to 6-DOF IMU titles if None.
            data_extractor: Function to extract data vector from a buffer item.
                            Default (None) assumes item is (timestamp, data) -> returns data.
                            Examples:
                            - Tuple (ts, data): lambda x: x[1] (Default)
                            - Dictionary item: lambda x: [x['val1'], x['val2']]
                            - Nested dict: lambda x: [x['imu']['acc_x'], x['imu']['acc_y']]
                            - Object: lambda x: [x.pos.x, x.pos.y]
            time_extractor: Function to extract timestamp from a buffer item.
                            Default (None) assumes item is (timestamp, data) -> returns timestamp.
                            Examples:
                            - Tuple (ts, data): lambda x: x[0] (Default)
                            - Dictionary item: lambda x: x['timestamp']
                            - Object: lambda x: x.header.stamp
                            Ignored if use_index_as_time is True.
            use_index_as_time: If True, use array index as x-axis instead of timestamp.
            **kwargs: Backward compatibility for label1, label2, etc.
        """
        self.buffers = buffers
        self.use_index_as_time = use_index_as_time

        # Default extractors for (timestamp, data) tuples
        if data_extractor is None:
            self.data_extractor = lambda x: x[1]
        else:
            self.data_extractor = data_extractor  # e.g. lambda x: x['data'], lambda x: x[2:5], etc.

        if time_extractor is None:
            self.time_extractor = lambda x: x[0]
        else:
            self.time_extractor = time_extractor

        # Handle labels
        if labels is None:
            self.labels = []
            # Try to find label1, label2, etc in kwargs for backward compatibility
            for i in range(len(buffers)):
                key = f"label{i+1}"
                if key in kwargs:
                    self.labels.append(kwargs[key])
                else:
                    self.labels.append(f"Buffer {i+1}")
        else:
            self.labels = labels
            # Ensure enough labels
            while len(self.labels) < len(self.buffers):
                self.labels.append(f"Buffer {len(self.labels)+1}")

        if subplot_titles is None:
            self.subplot_titles = ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)", "Wx (rad/s)", "Wy (rad/s)", "Wz (rad/s)"]
        else:
            self.subplot_titles = subplot_titles

        self.n_plots = len(self.subplot_titles)

        plt.ion()
        # Calculate grid size based on number of plots to avoid empty space
        if self.n_plots <= 3:
            cols = self.n_plots
            rows = 1
        elif self.n_plots == 4:
            cols = 2
            rows = 2
        else:
            # Attempt to make a somewhat square grid, but check screen aspect ratio preferences
            # For large N, we prefer wider grids
            cols = int(np.ceil(np.sqrt(self.n_plots)))
            if cols < 4 and self.n_plots > 9:
                cols = 4  # Ensure at least 4 cols for many plots

            rows = (self.n_plots + cols - 1) // cols

        # Ensure at least 1 row/col
        rows = max(1, rows)
        cols = max(1, cols)

        self.fig, self.axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        self.fig.suptitle(title)

        # Ensure axs is always iterable
        if not isinstance(self.axs, np.ndarray):
            self.axs = np.array([self.axs])

        self.lines = []  # List of lists. lines[subplot_index][buffer_index]

        # Colors/Styles
        styles = ["g-", "r--", "b-.", "k:", "m-", "c--"]

        # Flatten axs for easy iteration
        ax_flat = self.axs.flat

        for i in range(len(ax_flat)):
            ax = ax_flat[i]
            if i < self.n_plots:
                ax.set_title(self.subplot_titles[i])
                subplot_lines = []
                for j in range(len(self.buffers)):
                    style = styles[j % len(styles)]
                    (l,) = ax.plot([], [], style, label=self.labels[j], alpha=0.7)
                    subplot_lines.append(l)

                self.lines.append(subplot_lines)
                ax.legend(loc="upper right", fontsize="small")
                ax.grid(True)
            else:
                # Hide unused subplots
                ax.axis("off")

        self.fig.tight_layout()

    def update(self):
        if not self.buffers:
            return

        # Copy data to avoid modification during iteration
        data_snapshots = [list(b) for b in self.buffers]

        # Check if any buffer is empty (optional: we can plot what we have)
        # But if all are empty, return
        if all(len(d) == 0 for d in data_snapshots):
            return

        try:
            # Extract data and time
            processed_data = []
            times = []

            # Use the first non-empty buffer as time reference if not using index
            ref_time = 0.0
            if not self.use_index_as_time:
                for d in data_snapshots:
                    if len(d) > 0:
                        # Use the oldest timestamp in the window as reference (t=0)
                        # This ensures the plot starts at 0 and grows positively
                        ref_time = self.time_extractor(d[0])
                        break

            for d in data_snapshots:
                if len(d) == 0:
                    processed_data.append(None)
                    times.append(None)
                    continue

                raw_data = [self.data_extractor(x) for x in d]
                processed_data.append(np.array(raw_data))

                if self.use_index_as_time:
                    # Use simple 0-based index
                    times.append(np.arange(len(d)))
                else:
                    t = np.array([self.time_extractor(x) for x in d])
                    times.append(t - ref_time)

        except Exception as e:
            # print(f"Visualizer Error: {e}")
            return

        for i in range(self.n_plots):
            # self.lines[i] is list of lines for this subplot

            for j, (data, t) in enumerate(zip(processed_data, times)):
                if data is None or t is None:
                    continue

                val = self._get_col_data(data, i)
                if val is not None and len(val) > 0:
                    self.lines[i][j].set_data(t, val)

            ax = self.lines[i][0].axes
            ax.relim()
            ax.autoscale_view()

            # if not self.use_index_as_time:
            #     ax.set_xlim(-2.0, 0.1)
            # else: let autoscale handle it, it will grow from 0 to N

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _get_col_data(self, data, col_idx):
        if data.ndim == 1:
            if col_idx == 0:
                return data
            return None
        elif data.ndim > 1:
            if col_idx < data.shape[1]:
                return data[:, col_idx]
        return None


if __name__ == "__main__":
    import random
    import time
    from collections import deque

    print("Starting BufferVisualizer Demo...")

    # --- Scenario 1: Standard (timestamp, data_array) - 2 Buffers ---
    print("Running Scenario 1: Standard (timestamp, data_array) - 2 Buffers...")
    buf1 = deque(maxlen=100)
    buf2 = deque(maxlen=100)

    viz1 = BufferVisualizer(buf1, buf2, title="Scenario 1: 2 Buffers (6-DOF)")

    start_time = time.time()
    for i in range(30):
        t = start_time + i * 0.1
        data1 = np.random.rand(6)
        data2 = data1 + np.random.normal(0, 0.1, 6)

        buf1.append((t, data1))
        buf2.append((t, data2))

        viz1.update()
        plt.pause(0.05)

    plt.close(viz1.fig)

    # --- Scenario 2: 3 Buffers ---
    print("Running Scenario 2: 3 Buffers...")
    buf1.clear()
    buf2.clear()
    buf3 = deque(maxlen=100)

    viz2 = BufferVisualizer(buf1, buf2, buf3, labels=["GT", "Est 1", "Est 2"], title="Scenario 2: 3 Buffers")

    for i in range(30):
        t = start_time + i * 0.1
        data1 = np.random.rand(6)
        data2 = data1 + np.random.normal(0, 0.1, 6)
        data3 = data1 + np.random.normal(0, 0.2, 6)

        buf1.append((t, data1))
        buf2.append((t, data2))
        buf3.append((t, data3))

        viz2.update()
        plt.pause(0.05)

    plt.close(viz2.fig)

    # --- Scenario 3: Single Buffer ---
    print("Running Scenario 3: Single Buffer...")
    buf_single = deque(maxlen=30)

    viz3 = BufferVisualizer(buf_single, labels=["Single Source"], title="Scenario 3: Single Buffer")

    for i in range(50):
        t = start_time + i * 0.1
        data = np.random.rand(6)
        buf_single.append((t, data))
        viz3.update()
        plt.pause(0.05)

    plt.close(viz3.fig)

    # --- Scenario 4: No Timestamps (Index-based) ---
    print("Running Scenario 4: No Timestamps (Index-based)...")
    buf_no_ts1 = deque(maxlen=30)
    buf_no_ts2 = deque(maxlen=30)

    # Data is just the array, no tuple with timestamp
    # We use index as time, and align the latest data to 0
    viz4 = BufferVisualizer(
        buf_no_ts1,
        buf_no_ts2,
        labels=["Source A", "Source B"],
        title="Scenario 4: No Timestamps (Index-based)",
        use_index_as_time=True,
        data_extractor=lambda x: x,
    )  # Item is the data itself

    for i in range(50):
        # Just append data, no timestamp
        data1 = np.random.rand(6)
        data2 = data1 + np.random.normal(0, 0.1, 6)

        buf_no_ts1.append(data1)
        buf_no_ts2.append(data2)

        viz4.update()
        plt.pause(0.05)

    plt.close(viz4.fig)

    # --- Scenario 5: Dictionary Data with Custom Extractors ---
    print("Running Scenario 5: Dictionary Data with Custom Extractors...")
    dict_buf1 = deque(maxlen=50)
    dict_buf2 = deque(maxlen=50)

    # Data structure: {'timestamp': t, 'values': {'a': 1, 'b': 2}}
    # We want to plot 'a' and 'b'
    viz5 = BufferVisualizer(
        dict_buf1,
        dict_buf2,
        title="Scenario 5: Dictionary Data",
        subplot_titles=["Value A", "Value B"],
        # Extractor returns a list/array of values to plot for each subplot
        data_extractor=lambda x: [x["values"]["a"], x["values"]["b"]],
        # Extractor returns the timestamp
        time_extractor=lambda x: x["timestamp"],
    )

    for i in range(50):
        t = start_time + i * 0.1
        val1 = {"a": np.sin(i / 5.0), "b": np.cos(i / 5.0)}
        val2 = {"a": np.sin(i / 5.0) + 0.2, "b": np.cos(i / 5.0) - 0.2}

        dict_buf1.append({"timestamp": t, "values": val1})
        dict_buf2.append({"timestamp": t, "values": val2})

        viz5.update()
        plt.pause(0.05)

    plt.close(viz5.fig)

    print("Demo Done.")
