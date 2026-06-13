#!/usr/bin/env python3

# Copyright (C) 2025 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import psutil
import subprocess
from datetime import datetime
import numpy as np

import rclpy
import yasmin
from yasmin import StateMachine
from yasmin.blackboard import Blackboard
from yasmin_viewer import YasminViewerPub
from yasmin_ros import set_ros_loggers
from yasmin_ros.basic_outcomes import SUCCEED, TIMEOUT, CANCEL, ABORT
from yasmin_ros.yasmin_node import YasminNode

from vlm_rover_exploration.states.generate_map_image_state import GenerateMapImageState
from vlm_rover_exploration.states.drive_state import DriveState
from vlm_rover_exploration.states.llama_state import LlamaState
from vlm_rover_exploration.states.process_response_state import (
    ProcessResponseState,
    HAS_NEXT,
    HAS_NO_NEXT,
)
from vlm_rover_exploration.states.show_metrics_state import ShowMetricsState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


class MetricTracker:
    def __init__(self, node, blackboard):
        self.node = node
        self.blackboard = blackboard
        
        self.last_odom_est = None
        self.last_odom_real = None
        
        self.blackboard["total_distance_est_m"] = 0.0
        self.blackboard["total_distance_real_m"] = 0.0
        self.blackboard["points_count"] = 0
        self.blackboard["explored_area_m2"] = 0.0
        self.blackboard["start_time"] = self.node.get_clock().now()
        
        # Performance metrics
        self.blackboard["cpu_usage_samples"] = []
        self.blackboard["ram_usage_samples"] = []
        self.blackboard["gpu_usage_samples"] = []
        self.blackboard["vram_usage_samples"] = []
        
        # Duration breakdown
        self.blackboard["total_inference_time_s"] = 0.0
        self.blackboard["inference_times_s"] = []
        self.blackboard["total_navigation_time_s"] = 0.0

        self.node.create_subscription(Odometry, "/odom", self.odom_est_cb, 10)
        self.node.create_subscription(Odometry, "/odom_ground_truth", self.odom_real_cb, 10)
        
        # QoS for point cloud (transient local)
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.node.create_subscription(PointCloud2, "/cloud_map", self.cloud_cb, qos_profile=map_qos)

        # Timer to sample hardware metrics every 1.0 second
        self.metrics_timer = self.node.create_timer(1.0, self.sample_hardware_metrics)

    def sample_hardware_metrics(self):
        # CPU
        self.blackboard["cpu_usage_samples"].append(psutil.cpu_percent())
        # RAM
        self.blackboard["ram_usage_samples"].append(psutil.virtual_memory().used / (1024 * 1024))
        # GPU utilization (%)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                gpu_utils = [float(x) for x in result.stdout.strip().split('\n')]
                self.blackboard["gpu_usage_samples"].append(max(gpu_utils))
        except Exception:
            pass
        # VRAM usage (MB)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    used = float(line.strip())
                    self.blackboard["vram_usage_samples"].append(used)
                    break
        except Exception:
            pass

    def odom_est_cb(self, msg):
        curr = msg.pose.pose.position
        if self.last_odom_est is not None:
            d = np.sqrt((curr.x - self.last_odom_est.x)**2 + (curr.y - self.last_odom_est.y)**2)
            if d > 0.01:
                self.blackboard["total_distance_est_m"] += d
                self.last_odom_est = curr
        else:
            self.last_odom_est = curr

    def odom_real_cb(self, msg):
        curr = msg.pose.pose.position
        if self.last_odom_real is not None:
            d = np.sqrt((curr.x - self.last_odom_real.x)**2 + (curr.y - self.last_odom_real.y)**2)
            if d > 0.01:
                self.blackboard["total_distance_real_m"] += d
                self.last_odom_real = curr
        else:
            self.last_odom_real = curr

    def cloud_cb(self, msg):
        self.blackboard["points_count"] = msg.width * msg.height

def main():
    yasmin.YASMIN_LOG_INFO("yasmin_monitor_demo")

    # Initialize ROS 2
    rclpy.init()

    # Set ROS 2 logs
    set_ros_loggers()

    # Create a finite state machine (FSM)
    sm = StateMachine(outcomes=[SUCCEED])

    # Add states to the FSM
    sm.add_state(
        "GENERATING_MAP_IMAGE",
        GenerateMapImageState(),
        transitions={
            SUCCEED: "GENERATING_NEXT_WP",
            TIMEOUT: "GENERATING_MAP_IMAGE",
            CANCEL: "SHOW_METRICS",
        },
    )

    sm.add_state(
        "GENERATING_NEXT_WP",
        LlamaState(),
        transitions={
            SUCCEED: "PROCESSING_RESPONSE",
            ABORT: "GENERATING_MAP_IMAGE",
            CANCEL: "SHOW_METRICS",
        },
    )

    sm.add_state(
        "PROCESSING_RESPONSE",
        ProcessResponseState(),
        transitions={
            HAS_NEXT: "DRIVING_TO_WAYPOINT",
            HAS_NO_NEXT: "SHOW_METRICS",
            ABORT: "GENERATING_MAP_IMAGE", # If the label is not found
        },
    )

    sm.add_state(
        "DRIVING_TO_WAYPOINT",
        DriveState(),
        transitions={
            SUCCEED: "GENERATING_MAP_IMAGE",
            ABORT: "GENERATING_MAP_IMAGE", # If the drive fails
            CANCEL: "SHOW_METRICS",
        },
    )

    sm.add_state(
        "SHOW_METRICS",
        ShowMetricsState(),
        transitions={
            SUCCEED: SUCCEED,
        },
    )

    # Publish FSM information
    YasminViewerPub(sm, "YASMIN_MONITOR_DEMO")

    # Execute FSM
    try:
        bb = Blackboard()
        
        node = YasminNode.get_instance()
        tracker = MetricTracker(node, bb)
        
        bb["image_width_m"] = 20.0  # Width of the map in meters
        bb["image_height_m"] = 20.0  # Height of the map in meters
        bb["scale_factor"] = 10  # Scale factor for the map image
        bb["log_name"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        outcome = sm(bb)
        yasmin.YASMIN_LOG_INFO(outcome)
    except KeyboardInterrupt:
        if sm.is_running():
            sm.cancel_state()
    finally:
        # Shutdown ROS 2
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
