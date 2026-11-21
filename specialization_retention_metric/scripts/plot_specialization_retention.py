"""
plot_specialization_retention.py
Visualizes Tier 1 Specialization Retention Metrics as specified in Section 9 of the PRD.
Generates:
  1. Retention Trajectories over rounds (Weighted Mean & Worst-Client) across methods.
  2. Worst-Client Specialization Preservation comparison.
  3. Per-client scatter of Retention R vs. Label Entropy H_i & Aggregation Weight.
  4. Correlation scatter between Retention R and Client Accuracy on the Global Model.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

METHOD_COLORS = {
    'mi_iaa': '#1f77b4',   # Deep Blue
    'fedavg': '#d62728',   # Coral Red
    'fedcurv': '#2ca02c'   # Forest Green
}

METHOD_LABELS = {
    'mi_iaa': 'MI-IAA (Ours)',
    'fedavg': 'FedAvg',
    'fedcurv': 'FedCurv'
}


def load_latest_retention_data(results_dir, dataset_name):
    """
    Search for latest JSON and CSV retention log files for each method.
    Returns:
      summaries: dict {method: list_of_round_summaries}
      client_dfs: dict {method: dataframe_of_client_records}
    """
    logs_dir = os.path.join(results_dir, dataset_name, "logs")
    if not os.path.exists(logs_dir):
        print(f"Directory not found: {logs_dir}")
        return {}, {}

    summaries = {}
    client_dfs = {}

    for method in ['mi_iaa', 'fedavg', 'fedcurv']:
        json_pattern = os.path.join(logs_dir, f"specialization_retention_{method}_{dataset_name}_*.json")
        csv_pattern = os.path.join(logs_dir, f"specialization_retention_{method}_{dataset_name}_*.csv")

        json_files = sorted(glob.glob(json_pattern))
        csv_files = sorted(glob.glob(csv_pattern))

        if json_files:
            latest_json = json_files[-1]
            with open(latest_json, 'r', encoding='utf-8') as f:
                summaries[method] = json.load(f)
            print(f"Loaded {method} JSON from: {os.path.basename(latest_json)}")

        if csv_files:
            latest_csv = csv_files[-1]
            client_dfs[method] = pd.read_csv(latest_csv)
            print(f"Loaded {method} CSV from: {os.path.basename(latest_csv)}")

    return summaries, client_dfs


def plot_retention_trajectories(summaries, dataset_name, out_dir):
    """Plot Mean and Worst-Client Retention over communication rounds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for method, rounds_data in summaries.items():
        rounds = [r['round'] for r in rounds_data]
        weighted_r = [r['weighted_mean_retention'] for r in rounds_data]
        worst_r = [r['worst_client_retention'] for r in rounds_data]

        color = METHOD_COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method.upper())

        ax1.plot(rounds, weighted_r, marker='o', markersize=4, label=label, color=color, linewidth=2)
        ax2.plot(rounds, worst_r, marker='s', markersize=4, label=label, color=color, linewidth=2)

    ax1.set_title(f"Weighted Mean Retention R [t] ({dataset_name.upper()})", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Federated Round", fontsize=11)
    ax1.set_ylabel("Retention Ratio (R)", fontsize=11)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='Ideal Baseline (R=1.0)')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"Worst-Client Retention min_i R[i,t] ({dataset_name.upper()})", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Federated Round", fontsize=11)
    ax2.set_ylabel("Worst-Client Retention Ratio", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{dataset_name}_specialization_retention_trajectory.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot to: {save_path}")


def plot_entropy_and_weight_scatter(client_dfs, dataset_name, out_dir):
    """Scatter of final-round retention R vs. client label entropy and aggregation weight."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for method, df in client_dfs.items():
        if df.empty:
            continue
        last_round = df['round'].max()
        final_df = df[df['round'] == last_round]

        color = METHOD_COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method.upper())

        ax1.scatter(final_df['label_entropy'], final_df['R'], color=color, label=label, s=60, alpha=0.7)
        ax2.scatter(final_df['agg_weight'], final_df['R'], color=color, label=label, s=60, alpha=0.7)

    ax1.set_title(f"Final Round: Retention R vs. Label Entropy ({dataset_name.upper()})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Client Label Entropy H_i (nats)", fontsize=11)
    ax1.set_ylabel("Retention Ratio R", fontsize=11)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"Final Round: Retention R vs. Aggregation Weight ({dataset_name.upper()})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Client Aggregation Weight alpha_i", fontsize=11)
    ax2.set_ylabel("Retention Ratio R", fontsize=11)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{dataset_name}_retention_vs_entropy_weight.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Tier 1 Specialization Retention Metrics")
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'mnist'],
                        help="Dataset name to analyze.")
    parser.add_argument('--results_dir', type=str, default=None,
                        help="Path to results directory containing dataset logs.")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = args.results_dir or os.path.join(repo_root, "results")
    out_dir = os.path.join(results_dir, args.dataset, "plots")
    os.makedirs(out_dir, exist_ok=True)

    summaries, client_dfs = load_latest_retention_data(results_dir, args.dataset)

    if not summaries and not client_dfs:
        print(f"No retention logs found for {args.dataset} in {results_dir}. Run training with --eval_retention first.")
        return

    if summaries:
        plot_retention_trajectories(summaries, args.dataset, out_dir)
    if client_dfs:
        plot_entropy_and_weight_scatter(client_dfs, args.dataset, out_dir)


if __name__ == '__main__':
    main()
