# Copyright (C) 2025 Miguel Ángel González Santamarta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import json
import math
import numpy as np

from yasmin import State
from yasmin import Blackboard
from yasmin_ros.basic_outcomes import SUCCEED
import yasmin
from yasmin_ros.yasmin_node import YasminNode

class ShowMetricsState(State):
    def __init__(self) -> None:
        super().__init__(outcomes=[SUCCEED])

    def execute(self, blackboard: Blackboard) -> str:
        # Get metrics from blackboard
        area = blackboard["explored_area_m2"] if "explored_area_m2" in blackboard else 0.0
        points = blackboard["points_count"] if "points_count" in blackboard else 0
        dist_est = blackboard["total_distance_est_m"] if "total_distance_est_m" in blackboard else 0.0
        dist_real = blackboard["total_distance_real_m"] if "total_distance_real_m" in blackboard else 0.0
        
        # Calculate time
        start_time = blackboard["start_time"] if "start_time" in blackboard else 0.0
        current_time = YasminNode.get_instance().get_clock().now()
        duration = (current_time - start_time).nanoseconds / 1e9
        
        # Raw data extraction
        inference_times = blackboard["inference_times_s"] if "inference_times_s" in blackboard else []
        navigation_times = blackboard["navigation_times_s"] if "navigation_times_s" in blackboard else []

        cpu_samples = blackboard["cpu_usage_samples"] if "cpu_usage_samples" in blackboard else []
        gpu_samples = blackboard["gpu_usage_samples"] if "gpu_usage_samples" in blackboard else []
        ram_samples = blackboard["ram_usage_samples"] if "ram_usage_samples" in blackboard else []
        vram_samples = blackboard["vram_usage_samples"] if "vram_usage_samples" in blackboard else []
        
        route_history = blackboard["route_history"] if "route_history" in blackboard else []
        proximity_ranks = blackboard["proximity_ranks"] if "proximity_ranks" in blackboard else []

        model_path = os.environ.get("VLM_MODEL_CONFIG_PATH", "unknown")
        model_name = model_path.split("/")[-1].replace(".yaml", "")

        raw_metrics = {
            "area_m2": area,
            "dist_real_m": dist_real,
            "dist_est_m": dist_est,
            "points_count": points,
            "duration_s": duration,
            "inference_times_s": inference_times,
            "navigation_times_s": navigation_times,
            "cpu_usage_samples": cpu_samples,
            "gpu_usage_samples": gpu_samples,
            "ram_usage_samples": ram_samples,
            "vram_usage_samples": vram_samples,
            "route_history": route_history,
            "proximity_ranks": proximity_ranks
        }

        # Shared results file
        dir_name = f"{model_name}_{blackboard['log_name']}"
        results_file = f"{dir_name}/raw_metrics_{model_name}.json"
        
        all_data = {
            "model": model_name,
            "metrics": raw_metrics
        }

        # Save to file
        with open(results_file, 'w') as f:
            json.dump(all_data, f, indent=4)
        yasmin.YASMIN_LOG_INFO(f"Raw metrics saved to {results_file}")
        
        return SUCCEED
