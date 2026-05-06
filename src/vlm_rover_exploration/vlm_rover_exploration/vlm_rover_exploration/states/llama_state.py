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
import math

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

        robot_x, robot_y, _ = robot_position
        grid_mapping = blackboard["grid_mapping"]
        distances_text = ""
        if grid_mapping:
            distances_list = []
            for frontier_id, coords in grid_mapping.items():
                f_x = coords["x"]
                f_y = coords["y"]
                distance = math.hypot(robot_x - f_x, robot_y - f_y)
                distances_list.append(f"- ID {frontier_id}: {distance:.2f} meters")
            distances_text = "Euclidean distance from the robot to each frontier:\n" + "\n".join(distances_list) + "\n"

        radius = blackboard["image_width_m"] // 2
        # Determine if we should mention the blue line (second iteration onwards)
        has_history = "route_history" in blackboard and len(blackboard["route_history"]) > 0
        if has_history:
            trend_line_legend = "Magenta line: rover's recent path. The THICKER and DARKER end is the most recent position.\n"
            trend_line_instruction = "Use the magenta line to understand the rover's recent movement trend and follow the same logic if possible.\n"
            trend_prompt = "- movement_trend [1 sentence]: analyze the magenta line to determine the rover's recent direction of travel and exploration logic.\n                    "
            trend_global_instruction = " and the movement_trend"
            trend_local_instruction = " and movement_trend"
        else:
            trend_line_legend = ""
            trend_line_instruction = ""
            trend_prompt = ""
            trend_global_instruction = ""
            trend_local_instruction = ""

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
                    "Red dot labeled 'ROBOT': rover position\n"
                    f"Red arrow: rover orientation\n"
                    "White area: explored space\n"
                    "Gray area: unexplored space\n"
                    "Yellow area: obstacles. CRITICAL: ONLY yellow pixels represent obstacles. Gray area is simply unexplored space. You can ALWAYS navigate through white space to reach any frontier ID. Do NOT assume a frontier is unreachable or 'surrounded by obstacles' unless you see a clear YELLOW barrier blocking the path. Jagged gray edges are NOT obstacles.\n"
                    f"{trend_line_legend}"
                   #"Red 'X': recent failed navigation targets. CRITICAL: You MUST NOT select any frontier ID that is marked with or near a red 'X'. Avoid those areas entirely.\n"
                    "Numbered IDs: frontier cells. Valid exploration targets. CRITICAL: If an ID is visible, it MUST be considered a valid and necessary target. Even if an ID label appears to overlap with a white (explored) area, the underlying target point is ALWAYS on unexplored gray pixels. You MUST NOT discard an ID because you think it is already explored; if the label exists, it is a valid target.\n"
                    #f"{distances_text}"
                    f"Available frontiers IDs: {list(grid_mapping.keys())}\n"
                    "Analyze the map, the rover's trajectory, the obstacles if visible, and the distances to determine the best next frontier.\n"
                    f"{trend_line_instruction}"
                    "OUTPUT JSON:\n"
                    "- robot_localization [1 sentence]: ONLY describe the robot's position on the map and its orientation (using the crosshair labels TOP, BOTTOM, LEFT, RIGHT). Do NOT mention any IDs.\n" 
                    "- unexplored_area [1 sentence]: based on the robot_localization, identify and list ALL the different unexplored areas (gray color). Describe their locations relative to the robot. Do not mention any ID.\n"
                    f"{trend_prompt}- global_strategy [1-2 sentences]: based on the unexplored_area{trend_global_instruction}, determine ONE global exploration pattern to cover it efficiently (ej. spiral, zig-zag, etc.) and briefly justify why this pattern is chosen\n"
                    f"- local_strategy [1 sentence]: following the global strategy, decide if it is better to go to larger open spaces (vast unexplored fields for high range gain) or smaller gaps (isolated gray areas). If a small gap is close to the robot, prioritize clearing it to avoid backtracking. Briefly justify your choice.\n"
                    "- strategy_analysis [3-4 sentences]: combining the local_strategy, global_strategy, and distances, evaluate the 4 most relevant close frontiers to select the best one. Briefly justify the selected frontier and why you are discarding the others.\n"
                    "- target_label: [integer] ID of the selected frontier\n"
                    "- mission_complete: [boolean] Set to true ONLY if you believe all remaining frontiers are unreachable. Otherwise, set to false.\n"
            ),
            )
        ]

        # Attach image
        goal.images.append(self.cv_bridge.cv2_to_imgmsg(blackboard["map_image"]))

        # Sampling configuration and structured response schema
        goal.sampling_config.temp = 0.0
        goal.sampling_config.n_probs = 1
        properties = {
            "robot_situation": {"type": "string"},
            "unexplored_area": {"type": "string"},
        }
        required = ["robot_situation", "unexplored_area"]

        if has_history:
            properties["movement_trend"] = {"type": "string"}
            required.append("movement_trend")

        properties.update({
            "global_strategy": {"type": "string"},
            "local_strategy": {"type": "string"},
            "strategy_analysis": {"type": "string"},
            "target_label": {"type": "integer"},
            "mission_complete": {"type": "boolean"}
        })

        required.extend([
            "global_strategy", "local_strategy", "strategy_analysis", 
            "target_label", "mission_complete"
        ])

        grammar_dict = {
            "type": "object",
            "properties": properties,
            "required": required
        }

        goal.sampling_config.grammar_schema = json.dumps(grammar_dict)

        return goal

    def handle_result(
        self, blackboard: Blackboard, result: GenerateChatCompletions.Result
    ) -> str:
        try:
            response = result.choices[0].message.content
            blackboard["llama_response"] = response
            yasmin.YASMIN_LOG_INFO(f"LLaMA response:\n{response}")
            
            # Calculate Perplexity exclusively for strategy_analysis field
            logprobs_list = result.choices[0].logprobs
            if logprobs_list and len(logprobs_list) > 0:
                log_sum = 0.0
                num_tokens = 0
                in_strategy_analysis = False
                reconstructed_text = ""
                for token_prob_array in logprobs_list:
                    if hasattr(token_prob_array, 'data') and len(token_prob_array.data) > 0:
                        t_text = token_prob_array.data[0].token_text
                        
                        # Check for start key in reconstructed text
                        if not in_strategy_analysis:
                            if "strategy_analysis" in (reconstructed_text + t_text):
                                in_strategy_analysis = True
                        
                        # Check for end key in reconstructed text
                        if in_strategy_analysis:
                            if "target_label" in (reconstructed_text + t_text):
                                in_strategy_analysis = False
                                break
                            
                            cleaned_text = t_text.strip(' {}[]":,\n\r\t')
                            if cleaned_text:
                                p = token_prob_array.data[0].probability
                                
                                # The model provides log-probabilities (0.0 is max certainty, negative is lower)
                                # We sum them directly as log_probs.
                                if p <= 0:
                                    log_sum += p
                                else:
                                    # Fallback if somehow we get a raw probability > 0
                                    log_sum += math.log(max(p, 1e-10))
                                num_tokens += 1
                        
                        reconstructed_text += t_text
                
                if num_tokens > 0:
                    perplexity = math.exp(-log_sum / num_tokens)
                else:
                    perplexity = 1.0  # Default to 1.0 (perfectly certain) if no tokens found
                    yasmin.YASMIN_LOG_WARN(f"Perplexity calculation found 0 tokens in strategy_analysis. Defaulting to 1.0. Reconstructed text length: {len(reconstructed_text)}")
            else:
                perplexity = 0.0
                yasmin.YASMIN_LOG_WARN("No logprobs received, perplexity = 0")
                
            blackboard["perplexities"].append(perplexity)
            yasmin.YASMIN_LOG_INFO(f"Response Perplexity: {perplexity:.4f}")
            
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
            prompt_config_path = os.path.join(dir_name, "prompt_config.json")
            prompts_list = []
            if os.path.exists(prompt_config_path):
                try:
                    with open(prompt_config_path, "r") as f:
                        prompts_list = json.load(f)
                        if not isinstance(prompts_list, list):
                            prompts_list = [prompts_list]
                except Exception:
                    pass
            
            if len(prompts_list) < 2:
                prompts_list.append(prompt_data)
                with open(prompt_config_path, "w") as f:
                    json.dump(prompts_list, f, indent=4)
                
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
