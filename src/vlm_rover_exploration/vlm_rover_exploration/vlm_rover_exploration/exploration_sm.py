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

import rclpy
import yasmin
from yasmin import StateMachine
from yasmin.blackboard import Blackboard
from yasmin_viewer import YasminViewerPub
from yasmin_ros import set_ros_loggers
from yasmin_ros.basic_outcomes import SUCCEED, TIMEOUT, CANCEL, ABORT

from vlm_rover_exploration.states.generate_map_image_state import GenerateMapImageState
from vlm_rover_exploration.states.drive_state import DriveState
from vlm_rover_exploration.states.llama_state import LlamaState
from vlm_rover_exploration.states.process_response_state import (
    ProcessResponseState,
    HAS_NEXT,
    HAS_NO_NEXT,
)
from datetime import datetime

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
            CANCEL: SUCCEED,
        },
    )

    sm.add_state(
        "GENERATING_NEXT_WP",
        LlamaState(),
        transitions={
            SUCCEED: "PROCESSING_RESPONSE",
            ABORT: "GENERATING_MAP_IMAGE",
            CANCEL: SUCCEED,
        },
    )

    sm.add_state(
        "PROCESSING_RESPONSE",
        ProcessResponseState(),
        transitions={
            HAS_NEXT: "DRIVING_TO_WAYPOINT",
            HAS_NO_NEXT: SUCCEED,
            ABORT: "GENERATING_MAP_IMAGE", # If the label is not found
        },
    )

    sm.add_state(
        "DRIVING_TO_WAYPOINT",
        DriveState(),
        transitions={
            SUCCEED: "GENERATING_MAP_IMAGE",
            ABORT: "GENERATING_MAP_IMAGE", # If the drive fails
            CANCEL: SUCCEED,
        },
    )

    # Publish FSM information
    YasminViewerPub(sm, "YASMIN_MONITOR_DEMO")

    # Execute FSM
    try:
        bb = Blackboard()
        bb["image_width_m"] = 20.0  # Width of the map in meters
        bb["image_height_m"] = 20.0  # Height of the map in meters
        bb["scale_factor"] = 10  # Scale factor for the map image
        bb["log_name"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        outcome = sm(bb)
        yasmin.YASMIN_LOG_INFO(outcome)
    except KeyboardInterrupt:
        if sm.is_running():
            sm.cancel_state()

    # Shutdown ROS 2
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
