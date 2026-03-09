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


import yasmin
from yasmin_ros import ActionState
from yasmin_ros.basic_outcomes import SUCCEED
from yasmin.blackboard import Blackboard
from llama_msgs.msg import ChatMessage
from llama_msgs.action import GenerateChatCompletions
from cv_bridge import CvBridge
import os
import json
from datetime import datetime
import shutil

class LlamaState(ActionState):

    def __init__(self) -> None:
        self.cv_bridge = CvBridge()

        super().__init__(
            GenerateChatCompletions,
            "/llama/generate_chat_completions",
            self.create_llama_goal,
            result_handler=self.handle_result,
        )

    def create_llama_goal(self, blackboard: Blackboard) -> GenerateChatCompletions.Goal:
        robot_position = blackboard["robot_position"]
        yasmin.YASMIN_LOG_INFO(
            f"Creating LLaMA goal with robot position: {robot_position}"
        )
        radius = blackboard["image_width_m"] // 2

        goal = GenerateChatCompletions.Goal()
        goal.messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a Strategic Navigation System for a rover. Your goal is to explore the entire radius of the map efficiently using a frontier based approach. "
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"<__media__>\n"
                    "2D MAP:\n"
                    "White area: explored space\n"
                    "Gray area: unexplored space\n"
                    "Blue line: rover's recent path (NOTE: This line won't be visible in the first iteration)\n"
                    "Red dot: rover position\n"
                    f"Red arrow: rover orientation (yaw: {robot_position[2]} radians)\n"
                    "Numbered IDs: frontier cells. Targets\n"
                    "IMPORTANT: ID numbers do NOT indicate spatial proximity. For example, ID 1 and ID 2 may be far apart. "
                    "Always determine the actual position and adjacency of each ID by visually inspecting the map.\n"
                    "Analyze the map and determine the next best frontier target to ensure no area is left unexplored.\n\n"
                    "1. **Analyze**: Look at the Red Dot and the numbered frontier IDs on the map. Identify the unexplored areas.\n"
                    "2. **Global Strategy**: Decide a efficient strategy to explore the whole radius: Spiral, zig-zag, from center to border, etc.\n"
                    "3. **Local Strategy**: Decide if you should explore corners or gaps that should be closed now to avoid returning later or go to high-gain major areas.\n"
                    "4. **Execute**: Select the frontier ID that represents the best target for your chosen strategy. You MUST choose one of the numbered IDs visible on the map. Select a frontier that is close to the robot. If the blue line is visible, use it to understand the rover's recent movement trend and follow the same logic. For example, if the blue line has a trend to the right, mantain the direction to avoid sudden oscilations \n\n"
                    "OUTPUT JSON:\n"
                    "- description_of_the_map: VERY SHORT description. Answer ONLY these questions directly: 1. Where is the robot? (center, bottom, right, etc.) 2. Where is the robot facing? (the arrow is pointing to the right, left, top, bottom, etc.) 3. How are the surroundings? (bottom partially explored, top unexplored, etc.) 4. Which frontier cells are visible? 5. What is the last trend of the robot? (going right, left, top, bottom, etc.). DO NOT explain the map shape. DO NOT explain what colors mean. DO NOT add extra text.\n"
                    "- chosen_strategy: Explain with details the chosen global and local strategies to explore the radius of the map. Why is the chosen frontier ID the best option? If the blue line is visible, does it make sense with the last trend of the robot? \n"
                    "- target_label: [integer] (must be one of the frontier IDs on the map)\n"
                    "- target_yaw: [float (0 to 6.28)]\n"
                    "- is_fully_explored: [boolean]\n"
                ),
            )
        ]


        # Attach image
        goal.images.append(self.cv_bridge.cv2_to_imgmsg(blackboard["map_image"]))

        # Sampling configuration and structured response schema
        goal.sampling_config.temp = 0.3
        goal.sampling_config.grammar_schema = """{
            "type": "object",
            "properties": {
                "description_of_the_map": { 
                    "type": "string"},
                "chosen_strategy": { 
                    "type": "string"},
                "is_fully_explored": { 
                    "type": "boolean"},
                "target_label": { 
                    "type": "integer"},
                "target_yaw": { 
                    "type": "number"}
            },
            "required": ["description_of_the_map", "chosen_strategy", "is_fully_explored", "target_label", "target_yaw"]
        }"""

        return goal

    def handle_result(
        self, blackboard: Blackboard, result: GenerateChatCompletions.Result
    ) -> str:
        try:
            response = result.choices[0].message.content
            blackboard["llama_response"] = response
            yasmin.YASMIN_LOG_INFO(f"LLaMA response:\n{response}")
            
            # Save the JSON generated by the model in a single file            
            try:
                data = json.loads(response)
            except Exception:
                data = {"raw_response": response}
            
            # Get the path to the debug directory and make sure it exists
            dir_name = f"debug_{blackboard['log_name']}"
            os.makedirs(dir_name, exist_ok=True)
            
            # Save response
            with open(os.path.join(dir_name, "llama_responses.json"), "a") as f:
                f.write(json.dumps(data) + "\n")
                
            # Copy prompt and other context for debugging
            llama_goal = self.create_llama_goal(blackboard)
            prompt_data = {
                "system_prompt": llama_goal.messages[0].content,
                "user_prompt": llama_goal.messages[1].content,
                "temp": llama_goal.sampling_config.temp,
            }
            with open(os.path.join(dir_name, "prompt_config.json"), "w") as f:
                json.dump(prompt_data, f, indent=4)
                
            # Copy model configuration
            try:
                model_config_path = os.environ.get("VLM_MODEL_CONFIG_PATH")
                if model_config_path and os.path.exists(model_config_path):
                    shutil.copy(model_config_path, os.path.join(dir_name, "model_config.yaml"))
                else:
                    yasmin.YASMIN_LOG_WARN("VLM_MODEL_CONFIG_PATH not found or invalid.")
            except Exception as e:
                yasmin.YASMIN_LOG_WARN(f"Could not copy model configuration: {e}")

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Failed to process LLaMA result: {e}")
            blackboard["llama_response"] = "Error: No valid result from LLaMA."

        return SUCCEED
