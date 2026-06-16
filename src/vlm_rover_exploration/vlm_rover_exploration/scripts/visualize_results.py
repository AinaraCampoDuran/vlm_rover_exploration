#!/usr/bin/env python3

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_output_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


#!/usr/bin/env python3

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_output_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def make_composite_bar_chart(df, metric_configs, title, filename, model_order, model_colors):
    """Creates a single image with multiple bar chart subplots for grouped metrics."""
    n_metrics = len(metric_configs)
    
    # Enforce 2x2 grid for exactly 4 metrics
    if n_metrics == 4:
        cols = 2
        rows = 2
    else:
        cols = min(3, n_metrics)
        rows = int(np.ceil(n_metrics / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    colors = [model_colors[m] for m in model_order]

    for i, (col_name, m_title, m_unit) in enumerate(metric_configs):
        ax = axes[i]
        
        grouped = df.groupby('model')[col_name]
        means = grouped.mean().reindex(model_order).fillna(0)
        stds = grouped.std().reindex(model_order).fillna(0)
        
        ax.bar(
            model_order, means.values,
            yerr=stds.values, capsize=5,
            color=colors, edgecolor='white', linewidth=1.5
        )
        
        # Annotate exact values on top of bars
        for j, (val, std) in enumerate(zip(means.values, stds.values)):
            # Adjust annotation height so it doesn't overlap with error bar
            ax.text(j, val + std + means.max() * 0.05, f'{val:.2f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333')
                    
        ax.set_title(m_title, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel(m_unit, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelsize=11)
        sns.despine(ax=ax, top=True, right=True)

    # Hide unused subplots
    for j in range(len(metric_configs), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(filename, dpi=300)
    plt.close()


def make_composite_boxplot(df, metric_configs, title, filename, model_order, model_colors):
    """Creates a single image with multiple boxplot subplots for grouped metrics."""
    n_metrics = len(metric_configs)
    
    # Enforce 2x2 grid for exactly 4 metrics
    if n_metrics == 4:
        cols = 2
        rows = 2
    else:
        cols = min(3, n_metrics)
        rows = int(np.ceil(n_metrics / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    for i, (col_name, m_title, m_unit) in enumerate(metric_configs):
        ax = axes[i]
        
        # Boxplot showing distribution per model
        sns.boxplot(
            ax=ax, x='model', y=col_name, data=df,
            order=model_order, showfliers=False, width=0.5,
            boxprops=dict(alpha=0.4), palette=model_colors, hue='model', legend=False
        )
        # Overlay individual run dots
        sns.stripplot(
            ax=ax, x='model', y=col_name, data=df,
            order=model_order, size=6, jitter=True, alpha=0.7,
            edgecolor='black', linewidth=1, color='black'
        )
        
        ax.set_title(m_title, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel(m_unit, fontsize=11)
        ax.set_xlabel('')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelsize=11)
        sns.despine(ax=ax, top=True, right=True)

    # Hide unused subplots
    for j in range(len(metric_configs), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(filename, dpi=300)
    plt.close()


def load_strategy_data():
    """Search for llama_responses.json and load strategy counts."""
    files = glob.glob("*/llama_responses.json")
    if not files:
        for root, dirs, filenames in os.walk("."):
            if any(x in root for x in ['/build', '/install', '/log', '/src']):
                continue
            for f in filenames:
                if f == "llama_responses.json":
                    files.append(os.path.join(root, f))
    
    records = []
    for fpath in files:
        dir_path = os.path.dirname(fpath)
        bench_files = glob.glob(os.path.join(dir_path, "benchmark_results_*.json"))
        if not bench_files:
            continue
        
        model_name = "unknown"
        with open(bench_files[0], 'r') as bf:
            try:
                bdata = json.load(bf)
                model_name = bdata.get("model", "unknown")
            except:
                pass
                
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    strategy = data.get("global_strategy", "Unknown").strip().lower()
                    if strategy:
                        records.append({"model": model_name, "strategy": strategy})
                except Exception:
                    pass
    return pd.DataFrame(records) if records else None


def make_strategy_chart(df_strats, filename, model_order):
    """Stacked percentage bar chart showing the frequency of each strategy per model."""
    plt.figure(figsize=(10, 6))
    
    counts = df_strats.groupby(['model', 'strategy']).size().unstack(fill_value=0)
    counts = counts.reindex(model_order).fillna(0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    
    ax = percentages.plot(kind='bar', stacked=True, figsize=(10, 6), 
                          colormap='tab20', edgecolor='white', linewidth=1)
    
    plt.title('Global Strategy Selection Frequency (%)', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Percentage of Steps (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=0, fontsize=12, fontweight='bold')
    
    for c in ax.containers:
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=10, fontweight='bold', color='white')
        
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def load_data():
    files = glob.glob("*/benchmark_results_*.json")
    if not files:
        for root, dirs, filenames in os.walk("."):
            if any(x in root for x in ['/build', '/install', '/log', '/src']):
                continue
            for f in filenames:
                if f.startswith("benchmark_results_") and f.endswith(".json"):
                    files.append(os.path.join(root, f))
    if not files:
        print("No benchmark_results_*.json files found. Run benchmark.py first.")
        return None

    records = []
    for fpath in files:
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
                model_name = data.get("model", "unknown")
                metrics = data.get("metrics", {})
                if metrics:
                    record = {"model": model_name, "dir_path": os.path.dirname(fpath)}
                    for k, v in metrics.items():
                        if isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                record[f"{k}_{sub_k}"] = sub_v
                        else:
                            record[k] = v
                    records.append(record)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    return pd.DataFrame(records) if records else None


def main():
    df = load_data()
    if df is None or df.empty:
        return
    print(f"Loaded {len(df)} executions from {df['model'].nunique()} models.")

    base_dir = create_output_dir("benchmark_visualizations")
    sns.set_theme(style="whitegrid")

    model_order = sorted(df['model'].unique())
    palette = sns.color_palette("muted", n_colors=len(model_order))
    model_colors = dict(zip(model_order, palette))

    # ─── Group 1: Hardware Performance ──────────────────────────────
    perf_metrics = [
        ('cpu_percent_stats_avg',         'Avg CPU Usage',                       '%'),
        ('gpu_percent_stats_avg',         'Avg GPU Utilization',                 '%'),
        ('ram_mb_stats_avg',              'Avg RAM Usage',                       'MB'),
        ('vram_mb_stats_avg',             'Avg VRAM Usage',                      'MB'),
    ]
    perf_metrics = [m for m in perf_metrics if m[0] in df.columns]

    # ─── Group 2: Time & Latency ────────────────────────────────────
    time_metrics = [
        ('duration_s',                    'Total Mission Duration',              's'),
        ('navigation_time_s',             'Total Navigation Time',               's'),
        ('inference_time_s',              'Total Inference Time',                's'),
        ('inference_time_s_stats_avg',    'Avg Inference Latency / Step',        's'),
        ('navigation_time_s_stats_avg',   'Avg Navigation Time / Step',          's'),
    ]
    time_metrics = [m for m in time_metrics if m[0] in df.columns]

    # ─── Group 3: Exploration & Reliability ─────────────────────────
    expl_metrics = [
        ('area_m2',                       'Total Area Explored',                 'm²'),
        ('dist_real_m',                   'Total Distance Traveled',             'm'),
        ('coverage_ratio_m2_m',           'Coverage Ratio (Efficiency)',         'm² / m'),
        ('area_m2_per_step',              'Area Explored per Step',              'm² / step'),
        ('total_steps',                   'Total Decision Steps',                'Steps'),
        ('jump_distance_m_stats_avg',     'Avg Target Jump Distance',            'm'),
        ('proximity_rank_stats_avg',      'Avg Proximity Rank Selected',         'Rank'),
        ('nav_failures',                  'Navigation Failures',                 'Count'),
        ('error_odom',                    'Odometry Error',                      'm'),
    ]
    expl_metrics = [m for m in expl_metrics if m[0] in df.columns]

    print("Generating Composite Bar Charts (Averages)...")
    make_composite_bar_chart(df, perf_metrics, "Hardware & Latency Comparison", 
                             os.path.join(base_dir, "grouped_averages_performance.png"), 
                             model_order, model_colors)
    make_composite_bar_chart(df, time_metrics, "Time & Reliability Comparison", 
                             os.path.join(base_dir, "grouped_averages_time.png"), 
                             model_order, model_colors)
    make_composite_bar_chart(df, expl_metrics, "Exploration & Behavior Comparison", 
                             os.path.join(base_dir, "grouped_averages_exploration.png"), 
                             model_order, model_colors)

    print("Generating Composite Boxplots (Distributions per Run)...")
    make_composite_boxplot(df, perf_metrics, "Hardware & Latency Distributions", 
                           os.path.join(base_dir, "grouped_distributions_performance.png"), 
                           model_order, model_colors)
    make_composite_boxplot(df, time_metrics, "Time & Reliability Distributions", 
                           os.path.join(base_dir, "grouped_distributions_time.png"), 
                           model_order, model_colors)
    make_composite_boxplot(df, expl_metrics, "Exploration & Behavior Distributions", 
                           os.path.join(base_dir, "grouped_distributions_exploration.png"), 
                           model_order, model_colors)

    print(f"\nAll composite visualizations saved in '{base_dir}/'.")

    # ─── 4. Strategy Frequencies ──────────────────────────────────────
    print("Generating Strategy Frequency Chart...")
    df_strats = load_strategy_data()
    if df_strats is not None and not df_strats.empty:
        make_strategy_chart(df_strats, os.path.join(base_dir, "strategy_frequencies.png"), model_order)
    else:
        print("No strategy data found.")

if __name__ == "__main__":
    main()

