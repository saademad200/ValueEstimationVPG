#!/usr/bin/env python3
"""
Plot results for additional experiments.
Compares VPG variants: Adaptive, Large Critic, VPG+, and Baseline.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

EXPERIMENT_NAMES = {
    'vpg_adaptive': 'VPG Adaptive',
    'vpg_critic_128': 'VPG Critic 128x128',
    'vpg_critic_256': 'VPG Critic 256x256',
    'vpg_plus': 'VPG+',
    'vpg_valstep_50': 'VPG Baseline (vs=50)',
    'vpg_valstep_100': 'VPG (vs=100)',
    'vpg_mc': 'VPG Monte Carlo',
    'vpg_nstep5': 'VPG 5-step',
    'vpg_gae': 'VPG GAE',
    'vpg_gaenorm': 'VPG GAE (Norm)',
    'vpg_hybrid': 'VPG Hybrid (Ours)',
}


def parse_log_file(log_path: Path) -> dict:
    """Parse a log file to extract metrics."""
    rewards = []
    final_reward = None
    time_elapsed = None
    value_steps_used = None
    
    with open(log_path) as f:
        for line in f:
            # Match evaluation rewards
            if "Eval Reward:" in line:
                match = re.search(r'Eval Reward: ([\d.]+)', line)
                if match:
                    rewards.append(float(match.group(1)))
            
            # Match cumulative reward prints
            if "Cumulative Reward:" in line:
                match = re.search(r'Cumulative Reward: ([\d.]+)', line)
                if match:
                    rewards.append(float(match.group(1)))
            
            # Match time elapsed
            if "Time elapsed:" in line:
                match = re.search(r'Time elapsed: ([\d.]+)s', line)
                if match:
                    time_elapsed = float(match.group(1))
            
            # Match average value steps (for adaptive)
            if "Average value steps" in line:
                match = re.search(r'Average value steps per iteration: ([\d.]+)', line)
                if match:
                    value_steps_used = float(match.group(1))
    
    if rewards:
        final_reward = rewards[-1]
    
    return {
        'rewards': rewards,
        'final_reward': final_reward,
        'time_elapsed': time_elapsed,
        'value_steps_used': value_steps_used,
    }


def load_all_results(logs_dir: Path) -> pd.DataFrame:
    """Load results from all log files."""
    results = []
    
    # Regex to capture Env, Experiment Name, and Seed
    # Matches: {Env}_{ExpName}_s{Seed}.log
    # Example: Hopper-v4_vpg_adaptive_s0.log
    # We look for standard env names at the start
    filename_pattern = re.compile(r'^(Hopper-v4|Walker2d-v4|HalfCheetah-v4)_(.+)_s(\d+)\.log$')

    for log_file in logs_dir.glob("*.log"):
        match = filename_pattern.match(log_file.name)
        if not match:
            # Try fallback for old naming (no env prefix) if necessary, or skip
            # For this task, we assume new naming convention
            # print(f"Skipping non-matching file: {log_file.name}")
            continue
        
        env_name = match.group(1)
        exp_name = match.group(2)
        seed = int(match.group(3))
        
        metrics = parse_log_file(log_file)
        
        if metrics['final_reward'] is not None:
            results.append({
                'env': env_name,
                'experiment': exp_name,
                'display_name': EXPERIMENT_NAMES.get(exp_name, exp_name),
                'seed': seed,
                'final_reward': metrics['final_reward'],
                'time_elapsed': metrics['time_elapsed'],
                'value_steps_used': metrics['value_steps_used'],
            })
    
    return pd.DataFrame(results)


def plot_performance_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing final performance of all methods."""
    if len(df) == 0:
        print("No data to plot!")
        return
    
    summary = df.groupby('display_name').agg({
        'final_reward': ['mean', 'std']
    }).reset_index()
    summary.columns = ['method', 'mean', 'std']
    summary = summary.sort_values('mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(summary))
    bars = ax.barh(summary['method'], summary['mean'], xerr=summary['std'].fillna(0), 
                   capsize=3, color=colors)
    
    ax.set_xlabel('Final Return')
    ax.set_title('Performance Comparison of VPG Variants')
    
    # Add value labels
    for bar, mean in zip(bars, summary['mean']):
        ax.text(mean + 20, bar.get_y() + bar.get_height()/2, 
                f'{mean:.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_compute_efficiency(df: pd.DataFrame, output_dir: Path):
    """Plot performance vs. compute time."""
    if 'time_elapsed' not in df.columns or df['time_elapsed'].isna().all():
        print("No timing data available for compute efficiency plot")
        return
    
    summary = df.groupby('display_name').agg({
        'final_reward': 'mean',
        'time_elapsed': 'mean',
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(summary['time_elapsed'], summary['final_reward'], 
                        s=100, c=range(len(summary)), cmap='viridis')
    
    for _, row in summary.iterrows():
        ax.annotate(row['display_name'], 
                   (row['time_elapsed'], row['final_reward']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Final Return')
    ax.set_title('Compute Efficiency: Performance vs. Training Time')
    
    plt.tight_layout()
    output_path = output_dir / 'compute_efficiency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


    plt.close()


def plot_time_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing execution time of all methods."""
    if 'time_elapsed' not in df.columns or df['time_elapsed'].isna().all():
        print("No timing data available for time comparison plot")
        return
    
    summary = df.groupby('display_name').agg({
        'time_elapsed': 'mean'
    }).reset_index()
    summary = summary.sort_values('time_elapsed', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(summary))
    bars = ax.barh(summary['display_name'], summary['time_elapsed'], color=colors)
    
    ax.set_xlabel('Execution Time (seconds)')
    ax.set_title('Average Execution Time by Experiment')
    
    # Add value labels
    for bar, val in zip(bars, summary['time_elapsed']):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}s', va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'time_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_adaptive_value_steps(df: pd.DataFrame, output_dir: Path):
    """Show adaptive value steps statistics if available."""
    adaptive_data = df[df['experiment'] == 'vpg_adaptive']
    if len(adaptive_data) == 0 or adaptive_data['value_steps_used'].isna().all():
        print("No adaptive value steps data available")
        return
    
    # Compare with baseline (fixed 50 steps)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ['VPG Baseline\n(50 fixed)', 'VPG Adaptive\n(dynamic)']
    avg_steps = [50, adaptive_data['value_steps_used'].mean()]
    
    bars = ax.bar(methods, avg_steps, color=['#3498db', '#2ecc71'])
    
    ax.set_ylabel('Average Value Steps per Iteration')
    ax.set_title('Adaptive vs. Fixed Value Steps')
    
    for bar, val in zip(bars, avg_steps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'adaptive_value_steps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame) -> str:
    """Generate markdown summary table."""
    if len(df) == 0:
        return "No results available yet."
    
    summary = df.groupby('display_name').agg({
        'final_reward': ['mean', 'std', 'count'],
        'time_elapsed': 'mean',
    }).reset_index()
    summary.columns = ['Method', 'Mean Return', 'Std', 'Seeds', 'Time (s)']
    summary = summary.sort_values('Mean Return', ascending=False)
    
    lines = [
        "| Method | Return (mean ± std) | Seeds | Time (s) |",
        "|--------|---------------------|-------|----------|",
    ]
    
    for _, row in summary.iterrows():
        std = row['Std'] if pd.notna(row['Std']) else 0
        time_str = f"{row['Time (s)']:.0f}" if pd.notna(row['Time (s)']) else "N/A"
        lines.append(f"| {row['Method']} | {row['Mean Return']:.0f} ± {std:.0f} | {int(row['Seeds'])} | {time_str} |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Plot experiment results')
    parser.add_argument('--logs-dir', type=str, default='logs/experiments',
                       help='Directory containing log files')
    parser.add_argument('--output', type=str, default='results/experiments',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        print("Run experiments first: ./experiments/run_experiments.sh")
        return
    
    print(f"Loading results from: {logs_dir}")
    df = load_all_results(logs_dir)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"Found {len(df)} result entries")
    
    # Iterate over environments
    envs = df['env'].unique()
    print(f"Environments found: {envs}")

    for env in envs:
        print(f"\n=== Processing Environment: {env} ===")
        env_df = df[df['env'] == env].copy()
        env_output_dir = output_dir / env
        env_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots for this environment
        plot_performance_comparison(env_df, env_output_dir)
        plot_compute_efficiency(env_df, env_output_dir)
        plot_time_comparison(env_df, env_output_dir)
        plot_adaptive_value_steps(env_df, env_output_dir)
        
        # Print summary for this environment
        print(f"\n--- Summary Table for {env} ---")
        summary_table = generate_summary_table(env_df)
        print(summary_table)
        
        # Save summary
        summary_path = env_output_dir / 'summary.md'
        with open(summary_path, 'w') as f:
            f.write(f"# Experiment Results Summary: {env}\n\n")
            f.write(summary_table)
        print(f"Saved summary to: {summary_path}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
