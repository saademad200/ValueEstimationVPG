#!/usr/bin/env python3
"""
Plot results for Value Estimation VPG paper replication.
Generates charts comparing VPG vs PPO with different GAE lambda values.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paper's Table 2 reference values
PAPER_RESULTS = {
    ('VPG', 0.95, 'Hopper-v4'): (2601.57, 232.14),
    ('VPG', 1.0, 'Hopper-v4'): (2323.28, 436.24),
    ('PPO', 0.95, 'Hopper-v4'): (1965.29, 478.14),
    ('PPO', 1.0, 'Hopper-v4'): (1611.46, 541.32),
    ('VPG', 0.95, 'Walker2d-v4'): (3457.79, 646.70),
    ('VPG', 1.0, 'Walker2d-v4'): (3123.83, 987.58),
    ('PPO', 0.95, 'Walker2d-v4'): (2527.74, 507.40),
    ('PPO', 1.0, 'Walker2d-v4'): (1431.50, 612.17),
    ('VPG', 0.95, 'HalfCheetah-v4'): (4928.88, 807.04),
    ('VPG', 1.0, 'HalfCheetah-v4'): (4381.30, 222.15),
    ('PPO', 0.95, 'HalfCheetah-v4'): (4488.60, 2699.54),
    ('PPO', 1.0, 'HalfCheetah-v4'): (2604.08, 1237.80),
}


def load_results(csv_path: str) -> pd.DataFrame:
    """Load and preprocess results CSV."""
    df = pd.read_csv(csv_path)
    df = df[df['mean_return'] != 'NaN']
    df['mean_return'] = pd.to_numeric(df['mean_return'], errors='coerce')
    df['gae_lambda'] = pd.to_numeric(df['gae_lambda'], errors='coerce')
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std across seeds."""
    summary = df.groupby(['algorithm', 'gae_lambda', 'env']).agg({
        'mean_return': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['algorithm', 'gae_lambda', 'env', 'mean', 'std', 'n_seeds']
    return summary


def plot_comparison_bars(summary: pd.DataFrame, output_dir: Path):
    """Create bar chart comparing VPG vs PPO for each environment."""
    envs = summary['env'].unique()
    
    fig, axes = plt.subplots(1, len(envs), figsize=(4*len(envs), 5))
    if len(envs) == 1:
        axes = [axes]
    
    for ax, env in zip(axes, envs):
        env_data = summary[summary['env'] == env]
        
        x = np.arange(2)  # Two GAE lambda values
        width = 0.35
        
        for i, algo in enumerate(['VPG', 'PPO']):
            algo_data = env_data[env_data['algorithm'] == algo].sort_values('gae_lambda')
            means = algo_data['mean'].values
            stds = algo_data['std'].fillna(0).values
            
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, means, width, label=algo, yerr=stds, capsize=3)
        
        ax.set_xlabel('GAE λ')
        ax.set_ylabel('Return')
        ax.set_title(env.replace('-v4', ''))
        ax.set_xticks(x)
        ax.set_xticklabels(['0.95', '1.0'])
        ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'vpg_vs_ppo_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_paper_comparison(summary: pd.DataFrame, output_dir: Path):
    """Compare our results with paper's Table 2."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    
    for ax, env in zip(axes, envs):
        categories = []
        our_means, our_stds = [], []
        paper_means, paper_stds = [], []
        
        for algo in ['VPG', 'PPO']:
            for gae in [0.95, 1.0]:
                key = (algo, gae, env)
                cat = f"{algo}\nλ={gae}"
                categories.append(cat)
                
                # Our results
                row = summary[(summary['algorithm'] == algo) & 
                             (summary['gae_lambda'] == gae) & 
                             (summary['env'] == env)]
                if len(row) > 0:
                    our_means.append(row['mean'].values[0])
                    our_stds.append(row['std'].fillna(0).values[0])
                else:
                    our_means.append(0)
                    our_stds.append(0)
                
                # Paper results
                if key in PAPER_RESULTS:
                    paper_means.append(PAPER_RESULTS[key][0])
                    paper_stds.append(PAPER_RESULTS[key][1])
                else:
                    paper_means.append(0)
                    paper_stds.append(0)
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, our_means, width, label='Ours', yerr=our_stds, capsize=3, color='#2ecc71')
        ax.bar(x + width/2, paper_means, width, label='Paper', yerr=paper_stds, capsize=3, color='#3498db')
        
        ax.set_ylabel('Return')
        ax.set_title(env.replace('-v4', ''))
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8)
        ax.legend()
    
    plt.suptitle('Comparison with Paper Results (Table 2)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'paper_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_value_steps_ablation(df: pd.DataFrame, output_dir: Path):
    """Plot effect of different value steps (if data available)."""
    if 'value_steps' not in df.columns:
        return
    
    value_steps_data = df[df['algorithm'] == 'VPG']
    if value_steps_data['value_steps'].nunique() <= 1:
        return
    
    summary = value_steps_data.groupby(['value_steps', 'env']).agg({
        'mean_return': ['mean', 'std']
    }).reset_index()
    summary.columns = ['value_steps', 'env', 'mean', 'std']
    
    envs = summary['env'].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(4*len(envs), 5))
    if len(envs) == 1:
        axes = [axes]
    
    for ax, env in zip(axes, envs):
        env_data = summary[summary['env'] == env].sort_values('value_steps')
        ax.errorbar(env_data['value_steps'], env_data['mean'], 
                   yerr=env_data['std'].fillna(0), marker='o', capsize=3)
        ax.set_xlabel('Value Steps')
        ax.set_ylabel('Return')
        ax.set_title(env.replace('-v4', ''))
        ax.set_xscale('log')
    
    plt.suptitle('Effect of Value Update Steps on VPG Performance', fontsize=12, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'value_steps_ablation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_latex_table(summary: pd.DataFrame) -> str:
    """Generate LaTeX table matching paper's Table 2 format."""
    envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Algorithm & GAE $\lambda$ & Hopper & Walker & HalfCheetah \\",
        r"\hline",
    ]
    
    for algo in ['VPG', 'PPO']:
        for gae in [0.95, 1.0]:
            row = [algo, str(gae)]
            for env in envs:
                data = summary[(summary['algorithm'] == algo) & 
                              (summary['gae_lambda'] == gae) & 
                              (summary['env'] == env)]
                if len(data) > 0:
                    mean = data['mean'].values[0]
                    std = data['std'].values[0] if pd.notna(data['std'].values[0]) else 0
                    row.append(f"${mean:.2f} \\pm {std:.2f}$")
                else:
                    row.append("--")
            lines.append(" & ".join(row) + r" \\")
    
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Performance of VPG and PPO with different GAE factors.}",
        r"\label{tab:results}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Plot experiment results')
    parser.add_argument('--csv', type=str, default='results/summary_results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output', type=str, default='results/figures',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {args.csv}")
    df = load_results(args.csv)
    
    if len(df) == 0:
        print("No valid results found!")
        return
    
    print(f"Found {len(df)} result entries")
    
    summary = compute_summary(df)
    print("\n=== Summary Statistics ===")
    print(summary.to_string(index=False))
    
    # Generate plots
    print("\n=== Generating Plots ===")
    plot_comparison_bars(summary, output_dir)
    plot_paper_comparison(summary, output_dir)
    plot_value_steps_ablation(df, output_dir)
    
    # Generate LaTeX table
    latex = generate_latex_table(summary)
    latex_path = output_dir / 'table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {latex_path}")
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
