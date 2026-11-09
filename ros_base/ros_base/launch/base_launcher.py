import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from ros_base.utils.args_debug import add_debug_mode

"""
Base Launcher Orchestrator
---------------------------
A generic tool to manage multiple ROS 2 nodes/processes across different projects
using TMUX for session management and CPU affinity for performance optimization.
"""


class BaseLauncher:
    def __init__(self, config_path, enable_nodes=None, disable_nodes=None):
        self.enable_nodes = enable_nodes or []
        self.disable_nodes = disable_nodes or []
        # Resolve config path: check CWD first, then check relative to script directory
        self.config_path = self._resolve_config_path(config_path)

        if not self.config_path.exists():
            print(f"Error: Configuration file not found at {self.config_path}")
            sys.exit(1)

        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

        self._normalize_config()
        # self._validate_config() # Validation happens per session now

    def _resolve_config_path(self, path_str):
        """
        Smart path resolution.
        1. Try absolute or relative to CWD.
        2. If not found, try relative to the script's 'launch' directory.
        """
        path = Path(path_str)
        if path.exists():
            return path.resolve()

        # Try relative to script location
        script_dir = Path(__file__).parent
        path_alt = (script_dir / path_str).resolve()
        return path_alt

    def _normalize_config(self):
        """
        Normalize configuration to support list of sessions.
        If 'sessions' key exists, use it. Otherwise, wrap root config as a single session.
        """
        if "sessions" in self.cfg:
            self.sessions = self.cfg["sessions"]
        else:
            # Backward compatibility: Root is the session config
            # Ensure 'env' has 'session_name'
            if "env" in self.cfg and "session_name" not in self.cfg["env"]:
                self.cfg.setdefault("env", {})["session_name"] = "base_nodes"
            self.sessions = [self.cfg]

    def launch(self):
        """
        Orchestrates TMUX session and node execution.
        """
        print(f"Launching System from: {self.config_path}")

        import time

        for session_cfg in self.sessions:
            if "env" not in session_cfg:
                print(f"Error: Session config missing 'env' block.")
                continue

            env_cfg = session_cfg["env"]
            session_name = env_cfg.get("session_name", "base_session")

            # Kill existing session
            subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)
            print(f"Starting Session: [{session_name}]")

            nodes = session_cfg.get("nodes", [])
            active_nodes = []
            for node in nodes:
                name = node.get("name", "")
                if name in self.disable_nodes:
                    continue
                if name in self.enable_nodes:
                    active_nodes.append(node)
                    continue
                if node.get("autostart", True):
                    active_nodes.append(node)

            if not active_nodes:
                print(f"  -> No active nodes for session [{session_name}]. Skipping.")
                continue

            for i, node in enumerate(active_nodes):
                # 1. Prepare Paths & Commands
                ws_root = Path(env_cfg.get("workspace_root", "~/Project")).expanduser()
                project_path = ws_root / node.get("project", ".")

                # Silent environment setup
                setup_cmds = []
                if "ros_setup" in env_cfg:
                    ros_setup = Path(env_cfg["ros_setup"]).expanduser()
                    setup_cmds.append(f"source {ros_setup} > /dev/null 2>&1")

                if "conda_env" in env_cfg:
                    setup_cmds.append(f"conda activate {env_cfg['conda_env']} > /dev/null 2>&1")

                setup_cmds.append(f"cd {project_path}")

                setup_cmd_str = " && ".join(setup_cmds)

                # 2. Create Pane/Window
                if i == 0:
                    cmd = [
                        "tmux",
                        "new-session",
                        "-d",
                        "-s",
                        session_name,
                        "-n",
                        "MAIN",
                        "-P",
                        "-F",
                        "#{pane_id}",
                        "bash",
                    ]
                else:
                    cmd = ["tmux", "split-window", "-t", f"{session_name}:MAIN", "-P", "-F", "#{pane_id}", "bash"]

                pane_id = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
                subprocess.run(["tmux", "select-layout", "-t", f"{session_name}:MAIN", "tiled"])

                # 3. Initialize Environment
                full_init = f" {setup_cmd_str} && clear"
                subprocess.run(["tmux", "send-keys", "-t", pane_id, full_init, "C-m"])

                # Brief pause for shell to ready
                time.sleep(0.5)

                # 4. Run Node Command(s)
                commands = node["command"]
                if isinstance(commands, str):
                    commands = [commands]

                for cmd_str in commands:
                    # Only apply taskset if it looks like a single executable command,
                    # but for list of commands (like docker interaction) it might be tricky.
                    # We apply taskset to the first command if 'cpus' is specified.
                    # Actually, let's just apply taskset if 'cpus' is present and it's the first command?
                    # Or simpler: The user writes the command, we prefix taskset if needed.

                    final_cmd = cmd_str
                    if "cpus" in node and cmd_str == commands[0]:
                        final_cmd = f"taskset -c {node['cpus']} {cmd_str}"

                    subprocess.run(["tmux", "send-keys", "-t", pane_id, final_cmd, "C-m"])
                    # If sending multiple commands, we might need a small delay or just rely on input buffer
                    time.sleep(0.2)

                print(f"  -> [{i}] {node['name']} started.")

        self._print_usage_tip()

    def _print_usage_tip(self):
        print(f"\n Systems started in TMUX.")
        print(f"============================================================")
        for session in self.sessions:
            sname = session["env"].get("session_name", "unknown")
            print(f" Session: {sname}")
            print(f"   -> Attach: tmux attach -t {sname}")
        print(f"============================================================\n")


if __name__ == "__main__":
    parser = add_debug_mode(listen_port=9999)
    parser.add_argument("config", nargs="?", default="launch_cfg.yaml", help="Path to YAML config file")
    parser.add_argument("--enable", nargs="+", default=[], help="List of node names to enable explicitly")
    parser.add_argument("--disable", nargs="+", default=[], help="List of node names to disable explicitly")
    args = parser.parse_args()

    launcher = BaseLauncher(args.config, enable_nodes=args.enable, disable_nodes=args.disable)
    launcher.launch()
