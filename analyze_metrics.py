#!/usr/bin/env python3

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from llm_judge import LLMJudge


def main():
    directorios = [
        "debug_20260429_151016",
        "debug_20260429_152834",
    ]

    files = []
    for d in directorios:
        search_pattern = os.path.join(d, "benchmark_results_*.json")
        matched = glob.glob(search_pattern)
        if matched:
            files.extend(matched)
        else:
            print(f"No JSONs found in {d}")
    
    if not files:
        print("\nNo JSONs found in any of the provided directories.")
        return

    data_records = []
    judge = LLMJudge()

    
    for fpath in files:
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
                model_name = data.get("model", "unknown")
                metrics = data.get("metrics", {})
                
                if metrics:
                    record = {"model": model_name}
                    record.update(metrics)
                    
                    # LLM Judge Integration
                    dir_path = os.path.dirname(fpath)
                    responses_path = os.path.join(dir_path, "llama_responses.json")
                    if os.path.exists(responses_path):
                        print(f"Calling LLM Judge for {model_name} in {dir_path}...")
                        history = []
                        with open(responses_path, 'r') as rf:
                            for line in rf:
                                if line.strip():
                                    try:
                                        history.append(json.loads(line))
                                    except:
                                        pass
                        
                        # Collect images corresponding to history steps
                        image_paths = []
                        for i in range(len(history)):
                            img_p = os.path.join(dir_path, f"map_centered_{i}.png")
                            if os.path.exists(img_p):
                                image_paths.append(img_p)
                        
                        judge_result = judge.evaluate_mission(metrics, history, image_paths)
                        record.update({f"judge_{k}": v for k, v in judge_result["judge_scores"].items()})
                        record["high_variance"] = judge_result["high_variance_alert"]
                        
                        if judge_result["high_variance_alert"]:
                            print(f"!!! ALERT: High variance in Judge scores for {model_name} in {dir_path}")
                    
                    data_records.append(record)
                else:
                    print(f"No metrics found in {fpath}")
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    if not data_records:
        print("No metrics found in any of the provided JSON files.")
        return

    df = pd.DataFrame(data_records)
    
    print(df.head())    
    metrics_cols = [c for c in df.columns if c not in ['model']]
    
    summary = df.groupby('model')[metrics_cols].agg(['mean', 'std', 'count'])
    
    print(summary)
    
    summary.to_csv("metricas_resumen.csv")

    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Métricas de Exploración del Rover', fontsize=16)

    # Explored Area
    sns.boxplot(data=df, x='model', y='area_m2', ax=axes[0, 0], palette="Set2")
    sns.stripplot(data=df, x='model', y='area_m2', color=".3", ax=axes[0, 0], size=6, alpha=0.6)
    axes[0, 0].set_title('Explored Area (m²)')
    axes[0, 0].set_ylabel('Area (m²)')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # Mission Duration
    sns.boxplot(data=df, x='model', y='duration_s', ax=axes[0, 1], palette="Set2")
    sns.stripplot(data=df, x='model', y='duration_s', color=".3", ax=axes[0, 1], size=6, alpha=0.6)
    axes[0, 1].set_title('Mission Duration (s)')
    axes[0, 1].set_ylabel('Duration (s)')
    axes[0, 1].tick_params(axis='x', rotation=15)

    # Real Distance Travelled
    sns.boxplot(data=df, x='model', y='dist_real_m', ax=axes[1, 0], palette="Set2")
    sns.stripplot(data=df, x='model', y='dist_real_m', color=".3", ax=axes[1, 0], size=6, alpha=0.6)
    axes[1, 0].set_title('Real Distance Travelled (m)')
    axes[1, 0].set_ylabel('Distance (m)')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # Efficiency (Area / Distance)
    sns.boxplot(data=df, x='model', y='coverage_ratio_m2_m', ax=axes[1, 1], palette="Set2")
    sns.stripplot(data=df, x='model', y='coverage_ratio_m2_m', color=".3", ax=axes[1, 1], size=6, alpha=0.6)
    axes[1, 1].set_title('Coverage Efficiency (Area/Distance)')
    axes[1, 1].set_ylabel('Ratio (m²/m)')
    axes[1, 1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig('graficas_metricas_boxplots.png', dpi=300)
    
    # LLM Judge Scores Plot
    judge_cols = [c for c in df.columns if c.startswith('judge_')]
    if judge_cols:
        fig_j, axes_j = plt.subplots(2, 2, figsize=(14, 10))
        fig_j.suptitle('Análisis de LLM Judge (Averaged Scores)', fontsize=16)
        
        for i, col in enumerate(judge_cols[:4]):
            ax = axes_j[i//2, i%2]
            sns.boxplot(data=df, x='model', y=col, ax=ax, palette="Pastel1")
            sns.stripplot(data=df, x='model', y=col, color=".3", ax=ax, size=6, alpha=0.6)
            ax.set_title(col.replace('judge_', '').capitalize())
            ax.set_ylabel('Score (0-1)')
            ax.tick_params(axis='x', rotation=15)
            ax.set_ylim(0, 1.05)
            
        plt.tight_layout()
        plt.savefig('graficas_judge_scores.png', dpi=300)

    
    # Area vs Duration (Scatter)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x='duration_s', y='area_m2', hue='model', style='model', s=150, alpha=0.8, palette="Set1")
    plt.title('Área Explorada vs Duración')
    plt.xlabel('Duration (s)')
    plt.ylabel('Area (m²)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafica_area_vs_duracion.png', dpi=300)
    

if __name__ == "__main__":
    main()
