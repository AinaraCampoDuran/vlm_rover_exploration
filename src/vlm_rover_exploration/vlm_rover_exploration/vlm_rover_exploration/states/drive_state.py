# Copyright (C) 2025  Miguel Ángel González Santamarta

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


from yasmin_ros import ActionState
from yasmin.blackboard import Blackboard
from yasmin_ros.basic_outcomes import SUCCEED
from yasmin_ros.yasmin_node import YasminNode

from nav2_msgs.action import NavigateToPose


class DriveState(ActionState):

    def __init__(self) -> None:
        super().__init__(NavigateToPose, "/navigate_to_pose", self.create_nav_goal)

    def create_nav_goal(self, blackboard: Blackboard) -> NavigateToPose.Goal:
        goal = NavigateToPose.Goal()
        goal.pose = blackboard["waypoint"]
        return goal

    def execute(self, blackboard: Blackboard) -> str:
        node = YasminNode.get_instance()
        start_time = node.get_clock().now()
        outcome = super().execute(blackboard)
        end_time = node.get_clock().now()
        nav_duration = (end_time - start_time).nanoseconds / 1e9
        current_nav = blackboard["total_navigation_time_s"] if "total_navigation_time_s" in blackboard else 0.0
        blackboard["total_navigation_time_s"] = current_nav + nav_duration
        
        if "navigation_times_s" not in blackboard:
            blackboard["navigation_times_s"] = []
        blackboard["navigation_times_s"].append(nav_duration)
        
        if "route_history" in blackboard and len(blackboard["route_history"]) > 0:
            if outcome == SUCCEED:
                blackboard["route_history"][-1]["status"] = "success"
            else:
                blackboard["route_history"][-1]["status"] = "failed"
        
        return outcome
