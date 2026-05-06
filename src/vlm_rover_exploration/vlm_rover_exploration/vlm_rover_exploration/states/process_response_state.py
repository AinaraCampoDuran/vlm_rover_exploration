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


import cv2
import json
import math
import numpy as np
import copy
import yasmin
import os

from yasmin import State
from yasmin.blackboard import Blackboard
from yasmin_ros.basic_outcomes import ABORT

from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler

HAS_NEXT = "has_next"
HAS_NO_NEXT = "has_no_next"


class ProcessResponseState(State):

    def __init__(self, debug: bool = True) -> None:
        self.counter = 0
        self.debug = debug
        super().__init__([HAS_NEXT, HAS_NO_NEXT, ABORT])

    def execute(self, blackboard: Blackboard) -> str:
        grid_mapping = blackboard["grid_mapping"]

        if len(grid_mapping) == 0:
            if self.debug:
                self.save_debug_image(blackboard)
            return HAS_NO_NEXT

        response = json.loads(blackboard["llama_response"])
        
        # Check if Llama signaled mission completion
        if response.get("mission_complete", False):
            yasmin.YASMIN_LOG_INFO("VLM signaled mission completion.")
            if self.debug:
                self.save_debug_image(blackboard)
            return HAS_NO_NEXT

        target_label = str(response["target_label"])
        robot_position = blackboard["robot_position"]

        # Validate target_label BEFORE using it
        if target_label not in grid_mapping:
            yasmin.YASMIN_LOG_WARN(
                f"Label {target_label} not found in grid mapping. "
            )
            return ABORT

        waypoint = grid_mapping[target_label]

        # Compute target_yaw programmatically 
        dx = waypoint["x"] - robot_position[0]
        dy = waypoint["y"] - robot_position[1]
        target_yaw = math.atan2(dy, dx)

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(waypoint["x"])
        msg.pose.position.y = float(waypoint["y"])

        orientation = quaternion_from_euler(0.0, 0.0, target_yaw)
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]

        blackboard["waypoint"] = msg

        # Route history tracking
        if "route_history" not in blackboard:
            blackboard["route_history"] = []

        # Add robot's current position as the start of the route (first iteration)
        robot_position = blackboard["robot_position"]
        if len(blackboard["route_history"]) == 0:
            blackboard["route_history"].append(
                {"x": robot_position[0], "y": robot_position[1], "label": "start"}
            )

        # Append the new target waypoint to the route
        blackboard["route_history"].append(
            {"x": waypoint["x"], "y": waypoint["y"], "label": target_label}
        )

        if self.debug:
            self.save_debug_image(blackboard, target_label)

        return HAS_NEXT

    def save_debug_image(self, blackboard: Blackboard, target_label: str = None) -> None:
        map_image = copy.deepcopy(blackboard["map_image"])
        map_resolution = blackboard["map_resolution"]
        scale = blackboard["scale_factor"]
        center_x = blackboard["center_x"]
        center_y = blackboard["center_y"]
        pixels_per_meter = int(1.0 / map_resolution) * scale
        init_x, init_y = blackboard["initial_position"]

        # 1. Prepare route points for full history
        route_pts = []
        if "route_history" in blackboard:
            route = blackboard["route_history"]
            for point in route:
                px = int(center_x + ((point["x"] - init_x) * pixels_per_meter))
                py = int(center_y - ((point["y"] - init_y) * pixels_per_meter))
                route_pts.append((px, py))

            # Draw the current target waypoint (green, larger)
            if target_label and len(route_pts) > 0:
                current_target = route_pts[-1]
                cv2.circle(
                    map_image,
                    current_target,
                    3 * scale,
                    (0, 255, 0, 255),  # Green for current target
                    -1,
                )

        dir_name = f"debug_{blackboard['log_name']}"
        os.makedirs(dir_name, exist_ok=True)
        
        filename = os.path.join(dir_name, f"map_centered_{self.counter}.png")
        self.counter += 1
            
        cv2.imwrite(filename, map_image)

        # 3. Save FULL route image (Overlaying all points)
        if len(route_pts) > 1:
            full_route_image = copy.deepcopy(map_image)
            for i in range(1, len(route_pts)):
                cv2.line(
                    full_route_image,
                    route_pts[i - 1],
                    route_pts[i],
                    (255, 0, 255, 255),  # Magenta
                    max(1, scale // 2),
                    cv2.LINE_AA,
                )
            
            route_filename = os.path.join(dir_name, "full_route_history.png")
            cv2.imwrite(route_filename, full_route_image)

            # 4. Save pixel coordinates to JSON
            pixels_file = os.path.join(dir_name, "route_pixels.json")
            with open(pixels_file, "w") as f:
                json.dump(route_pts, f)
