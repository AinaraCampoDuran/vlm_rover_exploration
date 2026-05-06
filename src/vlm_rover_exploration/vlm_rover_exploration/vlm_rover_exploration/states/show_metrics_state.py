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
import time
import numpy as np
from yasmin import State
from yasmin import Blackboard
from yasmin_ros.basic_outcomes import SUCCEED
import yasmin

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
        duration = time.time() - start_time
        
        # Calculate derived metrics
        coverage_ratio = area / dist_real if dist_real > 0 else 0.0
        avg_speed = dist_real / duration if duration > 0 else 0.0
        
        # Calculate mean perplexity
        perplexities = blackboard["perplexities"] if "perplexities" in blackboard else []
        mean_perplexity = float(np.mean(perplexities)) if perplexities else 0.0
        
        model_path = os.environ.get("VLM_MODEL_CONFIG_PATH", "unknown")
        model_name = model_path.split("/")[-1]

        current_run = {
            "area_m2": area,
            "points": points,
            "dist_real_m": dist_real,
            "dist_est_m": dist_est,
            "duration_s": duration,
            "coverage_ratio_m2_m": coverage_ratio,
            "avg_speed_m_s": avg_speed,
            "mean_perplexity": mean_perplexity
        }

        # Shared results file
        dir_name = f"debug_{blackboard['log_name']}"
        results_file = f"{dir_name}/benchmark_results_{model_name}.json"
        
        all_data = {
            "model": model_name,
            "metrics": current_run
        }

        # Save to file
        with open(results_file, 'w') as f:
            json.dump(all_data, f, indent=4)
        yasmin.YASMIN_LOG_INFO(f"Metrics saved to {results_file}")

        # Also save to CSV
        import csv
        csv_file = f"{dir_name}/benchmark_results_{model_name}.csv"
        with open(csv_file, 'w', newline='') as f:
            fieldnames = sorted(current_run.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(current_run)
        yasmin.YASMIN_LOG_INFO(f"CSV results saved to {csv_file}")
        
        return SUCCEED
