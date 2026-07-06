#!/bin/bash

echo "Starting cleanup..."

ros2 daemon stop

# Kill ROS 2 nodes and launch files
pkill -9 -f ros2
pkill -9 -f exploration_sm
pkill -9 -f yasmin_viewer_node

# Kill Nav2 C++ nodes
pkill -9 -f bt_navigator
pkill -9 -f planner_server
pkill -9 -f controller_server
pkill -9 -f behavior_server
pkill -9 -f smoother_server
pkill -9 -f velocity_smoother
pkill -9 -f lifecycle_manager
pkill -9 -f ekf_node
pkill -9 -f robot_state_publisher
pkill -9 -f static_transform_publisher
pkill -9 -f rviz2

# Kill Gazebo / Ignition processes
pkill -9 -f "gz sim"
pkill -9 -f "ruby"
pkill -9 -f "ign"

# Kill RTAB-Map
pkill -9 -f rtabmap

# Kill LLaMA nodes
pkill -9 -f llava_node

# Stop ROS 2 daemon
ros2 daemon start

echo "Cleanup complete."
