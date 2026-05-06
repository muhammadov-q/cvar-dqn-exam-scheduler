"""Generate all figures from experiment results."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]

# Consistent color palette
AGENT_COLORS = {
    "Uniform": "#95a5a6",
    "MostUrgentFirst": "#7f8c8d",
    "LowestKnowledgeFirst": "#bdc3c7",
    "ValueIteration": "#2ecc71",
    "QLearning": "#3498db",
    "DQN": "#e74c3c",
    "CVaR-DQN": "#9b59b6",
}


def plot_main_comparison(results_dir: str, plots_dir: str) -> None:
    """Bar chart comparing all agents across configs."""
    path = os.path.join(results_dir, "results_all.csv")
    if not os.path.exists(path):
        logger.warning("results_all.csv not found, skipping main comparison plot")
        return

    df = pd.read_csv(path)
    metrics = ["mean_return", "mean_worst_exam", "failure_rate", "cvar_10"]
    titles = ["Mean Total Return", "Mean Worst Exam Score", "Failure Rate", "CVaR 10%"]

    for config_name in df["config"].unique():
        cfg_df = df[df["config"] == config_name]
        summary = cfg_df.groupby("agent").agg({m: "mean" for m in metrics}).reset_index()
        errors = cfg_df.groupby("agent").agg({m: "std" for m in metrics}).reset_index()

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Agent Comparison — {config_name.capitalize()} Config", fontsize=14)

        for ax, metric, title in zip(axes, metrics, titles):
            agents = summary["agent"]
            values = summary[metric]
            errs = errors[metric]
            colors = [AGENT_COLORS.get(a, "#34495e") for a in agents]

            ax.bar(range(len(agents)), values, yerr=errs, color=colors, capsize=3)
            ax.set_xticks(range(len(agents)))
            ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=8)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"comparison_{config_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved comparison_%s.png", config_name)


def plot_learning_curves(results_dir: str, plots_dir: str) -> None:
    """Learning curves with variance bands across seeds."""
    for config_name in ["small", "medium", "large"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Learning Curves — {config_name.capitalize()} Config")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (smoothed)")

        for agent_name, color in [("dqn", "#e74c3c"), ("cvar_dqn", "#9b59b6")]:
            all_curves: list[np.ndarray] = []
            for seed in SEEDS:
                path = os.path.join(results_dir, f"learning_curve_{agent_name}_{config_name}_seed{seed}.csv")
                if os.path.exists(path):
                    lc = pd.read_csv(path)["return"].values
                    # Smooth with rolling window
                    window = min(100, len(lc) // 10)
                    if window > 0:
                        smoothed = pd.Series(lc).rolling(window, min_periods=1).mean().values
                    else:
                        smoothed = lc
                    all_curves.append(smoothed)

            if not all_curves:
                continue

            min_len = min(len(c) for c in all_curves)
            curves = np.array([c[:min_len] for c in all_curves])
            mean_curve = curves.mean(axis=0)
            std_curve = curves.std(axis=0)

            episodes = np.arange(min_len)
            label = agent_name.upper().replace("_", "-")
            ax.plot(episodes, mean_curve, color=color, label=label)
            ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)

        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"learning_curves_{config_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved learning_curves_%s.png", config_name)


def plot_ablation(results_dir: str, plots_dir: str) -> None:
    """Ablation study bar charts."""
    path = os.path.join(results_dir, "ablation_results.csv")
    if not os.path.exists(path):
        logger.warning("ablation_results.csv not found, skipping ablation plot")
        return

    df = pd.read_csv(path)
    metrics = ["mean_total_reward", "cvar_10", "mean_worst_exam"]
    titles = ["Mean Return", "CVaR 10%", "Mean Worst Exam Score"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Ablation Study — Medium Config", fontsize=14)

    for ax, metric, title in zip(axes, metrics, titles):
        summary = df.groupby(["ablation", "agent"])[metric].mean().reset_index()
        pivot = summary.pivot(index="ablation", columns="agent", values=metric)
        pivot.plot(kind="bar", ax=ax, color=[AGENT_COLORS.get("DQN"), AGENT_COLORS.get("CVaR-DQN")])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ablation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation.png")


def plot_robustness(results_dir: str, plots_dir: str) -> None:
    """Robustness test comparison."""
    path = os.path.join(results_dir, "robustness_results.csv")
    if not os.path.exists(path):
        logger.warning("robustness_results.csv not found, skipping robustness plot")
        return

    df = pd.read_csv(path)
    metrics = ["mean_total_reward", "cvar_10", "mean_worst_exam"]
    titles = ["Mean Return", "CVaR 10%", "Mean Worst Exam Score"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Robustness Tests — DQN vs CVaR-DQN", fontsize=14)

    for ax, metric, title in zip(axes, metrics, titles):
        summary = df.groupby(["test", "agent"])[metric].mean().reset_index()
        pivot = summary.pivot(index="test", columns="agent", values=metric)
        pivot.plot(kind="bar", ax=ax, color=[AGENT_COLORS.get("CVaR-DQN"), AGENT_COLORS.get("DQN")])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "robustness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved robustness.png")


def plot_return_distributions(results_dir: str, plots_dir: str) -> None:
    """Histogram of return distributions for DQN vs CVaR-DQN."""
    path = os.path.join(results_dir, "results_all.csv")
    if not os.path.exists(path):
        return

    # This would require saving all returns; for now, plot from learning curves
    for config_name in ["small", "medium", "large"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"Return Distribution — {config_name.capitalize()}")
        ax.set_xlabel("Episode Return")
        ax.set_ylabel("Density")

        for agent_name, color in [("dqn", "#e74c3c"), ("cvar_dqn", "#9b59b6")]:
            # Use last 500 episodes from all seeds
            all_returns: list[float] = []
            for seed in SEEDS:
                lc_path = os.path.join(results_dir, f"learning_curve_{agent_name}_{config_name}_seed{seed}.csv")
                if os.path.exists(lc_path):
                    lc = pd.read_csv(lc_path)["return"].values
                    all_returns.extend(lc[-500:].tolist())

            if all_returns:
                label = agent_name.upper().replace("_", "-")
                ax.hist(all_returns, bins=50, alpha=0.5, color=color, label=label, density=True)

        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"return_dist_{config_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved return_dist_%s.png", config_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all plots from results")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plot_main_comparison(args.results_dir, plots_dir)
    plot_learning_curves(args.results_dir, plots_dir)
    plot_ablation(args.results_dir, plots_dir)
    plot_robustness(args.results_dir, plots_dir)
    plot_return_distributions(args.results_dir, plots_dir)

    logger.info("All plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
