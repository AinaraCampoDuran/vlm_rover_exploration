#!/usr/bin/env python3

import os
import glob
import json
import pandas as pd
import numpy as np
import math

def process_metrics(model_name, raw_metrics):
    area = raw_metrics.get("area_m2", 0.0)
    dist_real = raw_metrics.get("dist_real_m", 0.0)
    dist_estimated = raw_metrics.get("dist_est_m", 0.0)
    duration = raw_metrics.get("duration_s", 0.0)

    inference_times = raw_metrics.get("inference_times_s", [])
    # Calculate total inference time as sum of step inference times (fallback to raw_metrics field)
    inference_time = float(sum(inference_times)) if inference_times else raw_metrics.get("inference_time_s", 0.0)

    navigation_times = raw_metrics.get("navigation_times_s", [])
    # Calculate total navigation time as sum of step navigation times (fallback to raw_metrics field)
    navigation_time = float(sum(navigation_times)) if navigation_times else raw_metrics.get("navigation_time_s", 0.0)
    cpu_samples = raw_metrics.get("cpu_usage_samples", [])
    gpu_samples = raw_metrics.get("gpu_usage_samples", [])
    ram_samples = raw_metrics.get("ram_usage_samples", [])
    vram_samples = raw_metrics.get("vram_usage_samples", [])
    route_history = raw_metrics.get("route_history", [])
    proximity_ranks = raw_metrics.get("proximity_ranks", [])

    # Calculate navigation failures dynamically from route history status
    nav_failures = sum(1 for item in route_history if item.get("status") == "failed")
    # Fallback to hardcoded raw metrics field if available (for backward compatibility)
    if not nav_failures and "nav_failures" in raw_metrics:
        nav_failures = raw_metrics.get("nav_failures", 0)

    # Calculations
    total_steps = len(inference_times)
    coverage_ratio = area / dist_real if dist_real > 0 else 0.0
    area_per_step = area / total_steps if total_steps > 0 else 0.0
    error_odom = abs(dist_estimated - dist_real)
    
    def get_stats(data):
        if not data:
            return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "avg": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data))
        }

    predicted_labels = [str(item.get("label", "")) for item in route_history if str(item.get("label", "")) != "start"]
    
    jump_distances = []
    for i in range(1, len(route_history)):
        prev = route_history[i-1]
        curr = route_history[i]
        jump_dist = math.hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])
        jump_distances.append(jump_dist)

    complex_metrics = {
        "model_name": model_name,
        "area_m2": area,
        "dist_real_m": dist_real,
        "error_odom": error_odom,
        "total_steps": total_steps,
        "nav_failures": nav_failures,
        "duration_s": duration,
        "inference_time_s": inference_time,
        "navigation_time_s": navigation_time,
        "coverage_ratio_m2_m": coverage_ratio,
        "area_m2_per_step": area_per_step,
        "inference_time_s_stats": get_stats(inference_times),
        "navigation_time_s_stats": get_stats(navigation_times),
        "cpu_percent_stats": get_stats(cpu_samples),
        "gpu_percent_stats": get_stats(gpu_samples),
        "ram_mb_stats": get_stats(ram_samples),
        "vram_mb_stats": get_stats(vram_samples),
        "proximity_rank_stats": get_stats(proximity_ranks),
        "jump_distance_m_stats": get_stats(jump_distances),
        "predicted_labels": str(predicted_labels)
    }
    return complex_metrics

def main():
    # Search for all raw_metrics_*.json files dynamically in the workspace
    # It assumes the script is run from the workspace root (e.g. ros2_ws)
    files = glob.glob("*/raw_metrics_*.json")
    
    # If not found with the simple glob, try a recursive search avoiding large ROS 2 dirs
    if not files:
        for root, dirs, filenames in os.walk("."):
            if any(x in root for x in ['/build', '/install', '/log', '/src']):
                continue
            for f in filenames:
                if f.startswith("raw_metrics_") and f.endswith(".json"):
                    files.append(os.path.join(root, f))
                    
    if not files:
        print("\nNo JSONs found in the workspace.")
        return

    data_records = []
    for fpath in files:
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
                model_name = data.get("model", "unknown")
                raw_metrics = data.get("metrics", {})
                
                if raw_metrics:
                    metrics = process_metrics(model_name, raw_metrics)
                    
                    # Also save the computed complex metrics to a JSON for individual use
                    dir_path = os.path.dirname(fpath)
                    base_name = os.path.basename(fpath).replace("raw_metrics_", "benchmark_results_")
                    results_json = os.path.join(dir_path, base_name)
                    with open(results_json, 'w') as out_f:
                        json.dump({"model": model_name, "metrics": metrics}, out_f, indent=4)
                    
                    record = {"model": model_name, "dir_path": dir_path}
                    # Flatten nested dicts for pandas dataframe
                    for k, v in metrics.items():
                        if isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                record[f"{k}_{sub_k}"] = sub_v
                        else:
                            record[k] = v
                    
                    data_records.append(record)
                else:
                    print(f"No metrics found in {fpath}")
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    if not data_records:
        print("No metrics found in any of the provided JSON files.")
        return
        
    df = pd.DataFrame(data_records)
    
    for model, group in df.groupby('model'):
        model_dirs = group['dir_path'].tolist()
        final_metrics = {}
        
        for col in df.columns:
            if col in ['model', 'model_name', 'predicted_labels', 'dir_path']:
                continue
            if col.endswith('_avg') or col.endswith('_std') or col.endswith('_min') or col.endswith('_max'):
                continue
            
            std_val = float(group[col].std(ddof=0)) if len(group) > 1 else 0.0
            if math.isnan(std_val): std_val = 0.0
                
            final_metrics[col] = {
                "avg": float(group[col].mean()),
                "std": std_val,
                "min": float(group[col].min()),
                "max": float(group[col].max())
            }
            
        stats_prefixes = [c[:-4] for c in df.columns if c.endswith('_avg')]
        for prefix in stats_prefixes:
            std_val = float(group[prefix + '_avg'].std(ddof=0)) if len(group) > 1 else 0.0
            if math.isnan(std_val): std_val = 0.0
            
            avg_std_val = float(group[prefix + '_std'].mean())
            if math.isnan(avg_std_val): avg_std_val = 0.0
                
            final_metrics[prefix] = {
                "avg": float(group[prefix + '_avg'].mean()),
                "std": std_val,
                "min": float(group[prefix + '_min'].min()),
                "max": float(group[prefix + '_max'].max()),
                "avg_std": avg_std_val
            }
            
        out_data = {
            "model": model,
            "metrics": final_metrics,
            "directories_used": model_dirs
        }
        
        out_file = f"average_benchmark_{model}.json"
        with open(out_file, 'w') as f:
            json.dump(out_data, f, indent=4)
        print(f"Metrics for {model} saved to {out_file}")

if __name__ == "__main__":
    main()
