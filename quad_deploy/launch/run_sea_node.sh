#!/usr/bin/env bash
# Run after the launcher's Conda activation. Remove inherited ROS 1 paths,
# then load ROS 2 and the keyboard overlay in the actual node environment.
set -e

for sea_var in PATH PYTHONPATH LD_LIBRARY_PATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH PKG_CONFIG_PATH; do
    sea_filtered=""
    IFS=: read -r -a sea_entries <<< "${!sea_var}"
    for sea_entry in "${sea_entries[@]}"; do
        case "$sea_entry" in
            /opt/ros/noetic|/opt/ros/noetic/*|"") continue ;;
        esac
        sea_filtered="${sea_filtered:+${sea_filtered}:}${sea_entry}"
    done
    export "$sea_var=$sea_filtered"
done
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ROOT ROS_PACKAGE_PATH ROS_ETC_DIR ROSLISP_PACKAGE_DIRECTORIES
source /opt/ros/foxy/setup.bash
if [[ -f "$HOME/ros2_ws/install/local_setup.bash" ]]; then
    source "$HOME/ros2_ws/install/local_setup.bash"
fi

exec "$@"
