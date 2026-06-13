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
from yasmin_ros.yasmin_node import YasminNode       
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
import psutil
import subprocess
import threading
import time

class LlamaState(ActionState):

    def __init__(self) -> None:
        self.cv_bridge = CvBridge()

        super().__init__(
            GenerateChatCompletions,
            "/llama/generate_chat_completions",
            self.create_llama_goal,
            result_handler=self.handle_result,
        )
        self.node = YasminNode.get_instance()
        self.is_inferring = False
        self.inference_thread = None

    def _get_vram_usage(self, pid):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        p, mem = parts
                        if int(p.strip()) == pid:
                            return float(mem.strip())
        except Exception:
            pass
        return 0.0

    def _monitor_resources(self, blackboard):
        target_pid = None
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info['name']
                if name and ("llava_node" in name or "llama_node" in name):
                    target_pid = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if not target_pid:
            return

        if "cpu_usage_samples" not in blackboard:
            blackboard["cpu_usage_samples"] = []
        if "ram_usage_samples" not in blackboard:
            blackboard["ram_usage_samples"] = []
        if "vram_usage_samples" not in blackboard:
            blackboard["vram_usage_samples"] = []

        try:
            p = psutil.Process(target_pid)
            while self.is_inferring:
                try:
                    cpu = p.cpu_percent(interval=0.2)
                    ram_mb = p.memory_info().rss / (1024 * 1024)
                    vram_mb = self._get_vram_usage(target_pid)
                    
                    blackboard["cpu_usage_samples"].append(cpu)
                    blackboard["ram_usage_samples"].append(round(ram_mb, 2))
                    blackboard["vram_usage_samples"].append(vram_mb)
                except psutil.NoSuchProcess:
                    break
        except Exception:
            pass

    def create_llama_goal(self, blackboard: Blackboard) -> GenerateChatCompletions.Goal:
        self.is_inferring = True
        self.inference_thread = threading.Thread(target=self._monitor_resources, args=(blackboard,))
        self.inference_thread.start()

        blackboard["llama_start_time"] = self.node.get_clock().now()
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
        
        if "previous_global_strategy" in blackboard:
            previous_strategy = blackboard["previous_global_strategy"]
            strategy_instruction = f"- PREVIOUS STRATEGY: In the last step, you chose this strategy: '{previous_strategy}'. You should generally continue this strategy if it remains efficient, but you are free to change or break it if you justify the reason based on the current map state.\n"
            trend_global_instruction = "Evaluate if the PREVIOUS STRATEGY is still optimal. If so, continue it. If not, explicitly justify why you are changing it."
        else:
            strategy_instruction = ""
            trend_global_instruction = "Choose a pattern that makes logical sense given the robot's current position and heading."

        goal = GenerateChatCompletions.Goal()
     
        goal.messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a Strategic Navigation System for a planetary exploration rover. "
                    "Your ultimate objective is to map the entire area efficiently by expanding into unexplored territory. "
                    "Your specific goal is to systematically reduce a list of possible candidates until choosing a single target in the last step. "
                    "You must follow the entire reasoning process step-by-step to evaluate the options and choose the target."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"<__media__>\n"
                    "2D MAP EXPLANATION:\n"
                    "- Red dot labeled 'ROBOT': Current rover position.\n"
                    f"- Red arrow: Rover orientation/facing direction.\n"
                    "- White area: EXPLORED space."
                    "- Gray area: UNEXPLORED space. This is your target territory. "
                    "Your overall goal is to push the boundary of the white area outwards to reveal the gray area.\n"
                    "- Numbered IDs: Frontier cells located at the boundary between explored (white) and unexplored (gray) space. "
                    "Visiting a frontier ID expands the explored white area into the adjacent gray space.\n"
                    "- Crosshair (TOP, BOTTOM, LEFT, RIGHT): Global Reference Axes. CRITICAL RULE: ALL spatial descriptions MUST be absolutely relative to the center of this crosshair.\n"
                    f"{strategy_instruction}"
                    f"Euclidean distance from the robot to each frontier: {distances_text}. Do not estimate distances, use these values.\n"
                    "OUTPUT JSON:\n"
                    "- reasoning: [string] You MUST structure your reasoning exactly using this 8-step template (include the 'Step X:' labels). You MUST complete ALL 8 steps in full before closing this string field:\n"
                    "  Step 1: Describe ONLY the robot's position and orientation on the map using the crosshair (TOP, BOTTOM, LEFT, RIGHT).\n"
                    "  Step 2: INITIAL_CANDIDATES: [] (list all visible IDs).\n"
                    "  Step 3: Identify and list ALL the different unexplored areas (gray color) using the crosshair.\n"
                    f"  Step 4: Based on the unexplored areas from step 3, choose ONLY ONE global exploration pattern (spiral, perimeter, sweep, or zig-zag) and ONLY ONE direction (clockwise or counter-clockwise) to cover those areas efficiently. {trend_global_instruction}. Justify your decision.\n"
                    "  Step 5: Follow this exact logical sequence: 1) State the specific direction the robot must go next using the crosshair (e.g., TOP-RIGHT, BOTTOM-LEFT) to execute the global exploration pattern and direction chosen in step 4. 2) Identify which IDs are located in that specific direction. 3) Select a maximum of 3 of those IDs as GLOBAL_CANDIDATES: [list of 3 IDs or None].\n"
                    f"  Step 6: Select the single best candidate from GLOBAL_CANDIDATES. First, you MUST explicitly state the exact numerical distances (provided above) for EACH ID in GLOBAL_CANDIDATES. Then, evaluate them strictly against ALL three criteria: "
                    "1) Information Gain: how much new terrain will be discovered based on the size/openness of unexplored gray area near the ID, "
                    "2) Travel cost: compare their numerical distances, and "
                    f"3) Heading alignment: compare the target's location with the robot's orientation to evaluate turn cost; avoid very close curves. Select the best candidate according to these three criteria: GLOBAL_FINALIST: [single ID or None].\n"
                    "  Step 7: Scan for small, isolated gray gaps (such as any small patches identified in Step 3. A 'Hole' is defined strictly as a residual, dead-end gray patch that is deeply enclosed or surrounded on its sides by the white explored space. It must contain a very low concentration of frontier IDs (usually just 1). DO NOT evaluate distance or travel cost in this step; if a hole exists anywhere on the map, you MUST select its ID as LOCAL_FINALIST. If no candidates meet this strict dead-end criteria, you MUST set LOCAL_FINALIST: None.\n"
                    "  Step 8: Final Decision. Compare GLOBAL_FINALIST and LOCAL_FINALIST (if it exists). You must perform a cost-benefit analysis to decide if a local detour is justified. Weigh the immediate distance to the LOCAL_FINALIST against the risk of leaving an uncleaned area behind that might force a long backtracking journey later. This is the ONLY step where the final target is selected. Output: Winning ID: [single ID or None].\n"
                    "- global_strategy: [string] The exact pattern and direction chosen in Step 4. Format strictly as 'PATTERN DIRECTION' (e.g. 'spiral clockwise', 'perimeter counter-clockwise').\n"
                    "- target: [integer] ID of the selected frontier (None if no frontier is selected).\n"
                    "- mission_complete: [boolean] Set to true ONLY if INITIAL_CANDIDATES is None (meaning there are absolutely no unexplored areas left on the map and exploration is 100% finished). If there are any visible numbered IDs or active frontiers, the mission is NOT complete, and you MUST set mission_complete to false. Crucial: NEVER set mission_complete to true if a winning target ID was chosen!\n\n"
                )
            ),
        ]

        # Attach image
        goal.images.append(self.cv_bridge.cv2_to_imgmsg(blackboard["map_image"]))

        # Sampling configuration and structured response schema
        goal.sampling_config.temp = 0.2

        properties = {
            "reasoning": {"type": "string"},
            "global_strategy": {"type": "string"},
            "target": {"type": "integer"},
            "mission_complete": {"type": "boolean"}
        }
        required = ["reasoning", "global_strategy", "target", "mission_complete"]

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
        self.is_inferring = False
        if self.inference_thread is not None:
            self.inference_thread.join()

        if "llama_start_time" in blackboard:
            end_time = self.node.get_clock().now()
            inference_duration = (end_time - blackboard["llama_start_time"]).nanoseconds / 1e9
            current_inf = blackboard["total_inference_time_s"] if "total_inference_time_s" in blackboard else 0.0
            blackboard["total_inference_time_s"] = current_inf + inference_duration
            if "inference_times_s" in blackboard:
                blackboard["inference_times_s"].append(inference_duration)
            yasmin.YASMIN_LOG_INFO(f"Llama inference took {inference_duration:.2f} seconds")

        try:
            response = result.choices[0].message.content
            blackboard["llama_response"] = response
            yasmin.YASMIN_LOG_INFO(f"LLaMA response:\n{response}")
            
            # Save the JSON generated by the model in a single file            
            try:
                data = json.loads(response)
                # If there's extracted reasoning from the model's think block, populate it into the reasoning field
                if hasattr(result.choices[0].message, "reasoning_content") and result.choices[0].message.reasoning_content:
                    data["reasoning"] = result.choices[0].message.reasoning_content
                    # Write it back to the blackboard response as well so other states/logs see it
                    blackboard["llama_response"] = json.dumps(data)
                
                if "global_strategy" in data:
                    blackboard["previous_global_strategy"] = data["global_strategy"]
            except Exception:
                data = {"raw_response": response}
                if hasattr(result.choices[0].message, "reasoning_content") and result.choices[0].message.reasoning_content:
                    data["reasoning_content"] = result.choices[0].message.reasoning_content
            
            # Get the path to the debug directory and make sure it exists
            model_path = os.environ.get("VLM_MODEL_CONFIG_PATH", "unknown")
            model_name = model_path.split("/")[-1].replace(".yaml", "")
            dir_name = f"{model_name}_{blackboard['log_name']}"
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
