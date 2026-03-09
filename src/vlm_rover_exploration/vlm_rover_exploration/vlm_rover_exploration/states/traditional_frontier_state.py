import json
import math
import yasmin
from yasmin import State
from yasmin.blackboard import Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, CANCEL, ABORT

class TraditionalFrontierState(State):
    def __init__(self) -> None:
        super().__init__([SUCCEED, CANCEL, ABORT])

    def execute(self, blackboard: Blackboard) -> str:
        grid_mapping = blackboard.get("grid_mapping", {})
        
        if not grid_mapping:
            yasmin.YASMIN_LOG_INFO("No frontiers found, exploration complete.")
            blackboard["llama_response"] = json.dumps({
                "is_fully_explored": True,
                "target_label": "",
                "target_yaw": 0.0
            })
            return SUCCEED

        robot_position = blackboard.get("robot_position", (0.0, 0.0, 0.0))
        rx, ry, _ = robot_position

        nearest_label = None
        min_dist = float('inf')
        nearest_x = 0.0
        nearest_y = 0.0

        for label, coords in grid_mapping.items():
            cx = coords["x"]
            cy = coords["y"]
            dist = math.hypot(cx - rx, cy - ry)
            if dist < min_dist:
                min_dist = dist
                nearest_label = label
                nearest_x = cx
                nearest_y = cy

        if nearest_label is None:
            blackboard["llama_response"] = json.dumps({
                "is_fully_explored": True,
                "target_label": "",
                "target_yaw": 0.0
            })
            return SUCCEED

        # Calculate yaw pointing to the frontier
        target_yaw = math.atan2(nearest_y - ry, nearest_x - rx)

        response = {
            "is_fully_explored": False,
            "target_label": nearest_label,
            "target_yaw": target_yaw
        }
        
        blackboard["llama_response"] = json.dumps(response)
        yasmin.YASMIN_LOG_INFO(f"Traditional Frontier Selected: {nearest_label} at ({nearest_x:.2f}, {nearest_y:.2f})")

        return SUCCEED
