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

import math
import psutil
import subprocess
import threading
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
        self.node = YasminNode.get_instance()
        self.is_inferring = False
        self.inference_thread = None
        self.target_pid = None

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

    def _get_gpu_utilization(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                gpu_utils = [float(x) for x in result.stdout.strip().split('\n')]
                return max(gpu_utils)
        except Exception:
            pass
        return 0.0

    def _monitor_resources(self, blackboard):
        if not self.target_pid:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    name = proc.info['name']
                    if name and ("llava_node" in name or "llama_node" in name):
                        self.target_pid = proc.info['pid']
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        if not self.target_pid:
            return

        for key in ["cpu_usage_samples", "ram_usage_samples", "vram_usage_samples", "gpu_usage_samples"]:
            if key not in blackboard:
                blackboard[key] = []

        try:
            p = psutil.Process(self.target_pid)
            while self.is_inferring:
                try:
                    cpu = p.cpu_percent(interval=0.2)
                    ram_mb = p.memory_info().rss / (1024 * 1024)
                    vram_mb = self._get_vram_usage(self.target_pid)
                    gpu_util = self._get_gpu_utilization()
                    
                    blackboard["cpu_usage_samples"].append(cpu)
                    blackboard["ram_usage_samples"].append(round(ram_mb, 2))
                    blackboard["vram_usage_samples"].append(vram_mb)
                    blackboard["gpu_usage_samples"].append(gpu_util)
                except psutil.NoSuchProcess:
                    self.target_pid = None
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
            distances_list = [
                f"- ID {fid}: {math.hypot(robot_x - c['x'], robot_y - c['y']):.2f} meters" 
                for fid, c in grid_mapping.items()
            ]
            distances_text = "Euclidean distance from the robot to each frontier:\n" + "\n".join(distances_list) + "\n"


        if "previous_global_strategy" in blackboard:
            previous_strategy = blackboard["previous_global_strategy"]
            strategy_instruction = f"- PREVIOUS STRATEGY: In the last step, you chose this strategy: '{previous_strategy}'. You should generally continue this strategy if it remains efficient, but you are free to change or break it if you justify the reason based on the current map state.\n"
            trend_global_instruction = "Evaluate if the PREVIOUS STRATEGY is still optimal according to the current unexplored areas. If so, continue it. If not, justify why you are changing it."
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
                    "  Step 1: Describe ONLY the robot's position and orientation on the map using the crosshair.\n"
                    "  Step 2: INITIAL_CANDIDATES: [] (list all visible IDs).\n"
                    "  Step 3: Identify and list ALL the different unexplored areas (gray color) using the crosshair. Briefly describe their shape and size.\n"
                    f"  Step 4: Based on the unexplored areas from step 3, choose ONLY ONE global exploration pattern (spiral, perimeter, sweep, or zig-zag) and ONLY ONE direction (clockwise or counter-clockwise) to cover those areas efficiently. {trend_global_instruction}. Explain how the chosen pattern and direction will help to cover the unexplored areas in an efficient way.\n"
                    f"  Step 5: Follow this exact logical sequence: 1) State the specific direction the robot must go next using the crosshair to execute the global pattern and direction chosen in step 4. 2) Identify and list ALL the IDs that are located in that specific direction. 3) Select up to 3 of those IDs as GLOBAL_CANDIDATES. GLOBAL_CANDIDATES: [list of up to 3 IDs or None].\n"
                    f"  Step 6: Select the single best candidate from GLOBAL_CANDIDATES. First, you MUST explicitly state the exact numerical distances (provided above) for EACH ID in GLOBAL_CANDIDATES. Then, evaluate them strictly against ALL three criteria: "
                    "1) Information Gain: how much new terrain will be discovered based on the size/openness of unexplored gray area near the ID, "
                    "2) Travel cost: compare their numerical distances, and "
                    f"3) Heading alignment: compare the target's location with the robot's orientation to evaluate turn cost. Select the best candidate according to these three criteria. GLOBAL_FINALIST: [single ID or None].\n"
                    "  Step 7: Scan carefully for any 'holes', 'residual gray patches', or 'unexplored strips' that break the global pattern and direction. A true hole or residual patch MUST be an unexplored gray area that is disconnected from the main continuous unexplored mass. It may be completely surrounded by explored white space OR it may touch the outer perimeter of the circular map boundary. Normal corners or edges along the main continuous unexplored area are NOT holes. First, explicitly identify and list ALL the IDs that fit the descriptions. Then, compare their exact numerical distances (provided above) and select the CLOSEST one as the LOCAL_FINALIST. If none exist, you MUST set LOCAL_FINALIST: None.\n"
                    "  Step 8: Final Decision. Compare GLOBAL_FINALIST and LOCAL_FINALIST (if it exists). You must perform a cost-benefit analysis to decide if a local detour is justified. You are ALLOWED to completely break and override the global strategy (the pattern and direction chosen in Step 4) if choosing the LOCAL_FINALIST is worth it to clean up the area now and avoid a long, unnecessary backtracking journey in the future. Weigh the distances carefully. This is the ONLY step where the final target is selected. Output: Winning ID: [single ID or None].\n"
                    "- global_strategy: [string] The exact pattern and direction chosen in Step 4. Format strictly as 'PATTERN DIRECTION' (e.g. 'spiral clockwise', 'perimeter counter-clockwise').\n"
                    "- target: [integer] ID of the selected frontier (None if no frontier is selected).\n"
                                    )
            ),
        ]

        # Attach image
        goal.images.append(self.cv_bridge.cv2_to_imgmsg(blackboard["map_image"]))

        # Sampling configuration and structured response schema
        goal.sampling_config.temp = 0.0

        properties = {
            "reasoning": {"type": "string"},
            "global_strategy": {"type": "string"},
            "target": {"type": "integer"},
        }
        required = ["reasoning", "global_strategy", "target"]

        grammar_dict = {
            "type": "object",
            "properties": properties,
            "required": required
        }

        goal.sampling_config.grammar_schema = json.dumps(grammar_dict)

        blackboard["prompt_data"] = {
            "system_prompt": goal.messages[0].content,
            "user_prompt": goal.messages[1].content,
            "temp": goal.sampling_config.temp,
        }

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
            reasoning_content = getattr(result.choices[0].message, "reasoning_content", None)
            try:
                # Try direct parse first, then extract JSON from mixed content
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    if start_idx != -1 and end_idx > start_idx:
                        data = json.loads(response[start_idx:end_idx+1])
                    else:
                        raise
                
                if reasoning_content:
                    data["reasoning_content"] = reasoning_content
                
                # Unconditionally update the blackboard with the clean extracted JSON
                blackboard["llama_response"] = json.dumps(data)
                
                if "global_strategy" in data:
                    blackboard["previous_global_strategy"] = data["global_strategy"]
            except Exception:
                data = {"raw_response": response}
                if reasoning_content:
                    data["reasoning_content"] = reasoning_content
                blackboard["llama_response"] = json.dumps(data)
            
            model_path = os.environ.get("VLM_MODEL_CONFIG_PATH", "unknown")
            model_name = os.path.basename(model_path).replace(".yaml", "")
            log_name = blackboard["log_name"] if "log_name" in blackboard else "log"
            dir_name = f"{model_name}_{log_name}"
            os.makedirs(dir_name, exist_ok=True)
            
            # Save response
            with open(os.path.join(dir_name, "llama_responses.json"), "a") as f:
                f.write(json.dumps(data) + "\n")
                
            # Copy prompt and other context for debugging (up to 2 prompts: initial and with previous strategy)
            prompts_saved = blackboard["prompts_saved_count"] if "prompts_saved_count" in blackboard else 0
            if prompts_saved < 2:
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
                    prompts_list.append(blackboard["prompt_data"] if "prompt_data" in blackboard else {})
                    with open(prompt_config_path, "w") as f:
                        json.dump(prompts_list, f, indent=4)
                    blackboard["prompts_saved_count"] = len(prompts_list)
                    
                    # Copy model configuration only once when creating the first prompt
                    if len(prompts_list) == 1:
                        if model_path != "unknown" and os.path.exists(model_path):
                            shutil.copy(model_path, os.path.join(dir_name, "model_config.yaml"))
                        else:
                            yasmin.YASMIN_LOG_WARN("VLM_MODEL_CONFIG_PATH not found or invalid.")

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Failed to process LLaMA result: {e}")
            blackboard["llama_response"] = "Error: No valid result from LLaMA."

        return SUCCEED
