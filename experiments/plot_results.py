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
    "SC-CVaR-DQN": "#16a085",
}


def _max_achievable_psf(psf_list: list[float]) -> float:
    """Worst-subject failure rate among ACHIEVABLE subjects.

    Excludes subjects that are universally-100%-failing (env-infeasible
    even for the optimal policy). Returns 0 if all subjects are
    infeasible (degenerate case).
    """
    if not psf_list:
        return float("nan")
    # Subjects that are not universally infeasible (failure < 0.999)
    achievable = [p for p in psf_list if p < 0.999]
    return max(achievable) if achievable else 0.0


def plot_main_comparison(results_dir: str, plots_dir: str) -> None:
    """Per-config agent comparison: horizontal bars sorted by CVaR_10.

    Three panels: Mean Return, CVaR_10 (risk), and Worst-Achievable
    Per-Subject Failure Rate. The Failure Rate metric used in earlier
    versions of this report was the fraction of episodes with ANY
    sub-threshold subject; that metric is always 1.0 because every
    config has at least one environment-infeasible subject (Value
    Iteration also fails it 100%). We replace it with the worst
    failure rate among *achievable* subjects, which actually varies.
    """
    path = os.path.join(results_dir, "results_all.csv")
    if not os.path.exists(path):
        logger.warning("results_all.csv not found, skipping main comparison plot")
        return

    df = pd.read_csv(path)

    # Parse per-subject failure rate into a list per row.
    def parse_psf(s: str) -> list[float]:
        if pd.isna(s) or s == "":
            return []
        return [float(x) for x in str(s).split(";")]

    df["psf_list"] = df["per_subject_failure_rate"].apply(parse_psf)
    df["worst_achievable_psf"] = df["psf_list"].apply(_max_achievable_psf)

    rl_agents = {"DQN", "CVaR-DQN", "SC-CVaR-DQN", "ValueIteration", "QLearning"}

    for config_name in df["config"].unique():
        cfg_df = df[df["config"] == config_name]
        agents = cfg_df["agent"].unique().tolist()

        means = cfg_df.groupby("agent").agg({
            "mean_return": "mean",
            "cvar_10": "mean",
            "worst_achievable_psf": "mean",
        })
        stds = cfg_df.groupby("agent").agg({
            "mean_return": "std",
            "cvar_10": "std",
            "worst_achievable_psf": "std",
        })

        # Sort agents by CVaR_10 ascending (worst at top, best at bottom)
        order = means["cvar_10"].sort_values().index.tolist()
        means = means.loc[order]
        stds = stds.loc[order]

        fig, axes = plt.subplots(1, 3, figsize=(16, 0.5 + 0.5 * len(order)))
        fig.suptitle(f"Agent Comparison — {config_name.capitalize()} Config", fontsize=14)

        panels = [
            ("mean_return", "Mean Return", axes[0]),
            ("cvar_10", "CVaR$_{10\\%}$ (worst-case)", axes[1]),
            ("worst_achievable_psf", "Worst Achievable Subject Failure", axes[2]),
        ]

        for col, title, ax in panels:
            colors = [AGENT_COLORS.get(a, "#34495e") for a in means.index]
            edge_colors = ["black" if a in rl_agents else "none" for a in means.index]
            linewidths = [1.2 if a in rl_agents else 0 for a in means.index]
            ax.barh(
                range(len(means)), means[col], xerr=stds[col].fillna(0),
                color=colors, edgecolor=edge_colors, linewidth=linewidths,
                capsize=3,
            )
            ax.set_yticks(range(len(means)))
            ax.set_yticklabels(means.index, fontsize=9)
            ax.set_title(title, fontsize=11)
            ax.grid(axis="x", alpha=0.3)
            ax.invert_yaxis()

        axes[2].set_xlim(0, 1.05)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"comparison_{config_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved comparison_%s.png", config_name)


def plot_radar(results_dir: str, plots_dir: str) -> None:
    """Radar (spider) chart per config comparing DQN, CVaR-DQN, SC-CVaR-DQN.

    Axes:
      - Mean Return (higher = better)
      - CVaR_10 (higher = better)
      - One axis per subject: pass rate (1 - per-subject failure rate)

    All values are normalized to [0, 1] within each config using the
    range across the three agents on that config (so the radar reflects
    *relative* performance, with the worst agent's axis = 0 and the
    best = 1). The shape of each polygon makes per-subject trade-offs
    immediately legible.
    """
    path = os.path.join(results_dir, "results_all.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    keep = ["DQN", "CVaR-DQN", "SC-CVaR-DQN"]
    df = df[df["agent"].isin(keep)].copy()
    if df.empty:
        return

    def parse_psf(s):
        if pd.isna(s) or s == "":
            return []
        return [float(x) for x in str(s).split(";")]

    df["psf_list"] = df["per_subject_failure_rate"].apply(parse_psf)

    config_order = [c for c in ["small", "medium", "large"] if c in df["config"].unique()]
    n_configs = len(config_order)
    fig, axes = plt.subplots(
        1, n_configs, figsize=(7.5 * n_configs, 7.5),
        subplot_kw=dict(projection="polar"), squeeze=False,
    )

    LINESTYLES = {"DQN": "-", "CVaR-DQN": "-", "SC-CVaR-DQN": "-"}
    LINEWIDTHS = {"DQN": 2.5, "CVaR-DQN": 2.5, "SC-CVaR-DQN": 3.0}

    for ax, config_name in zip(axes[0], config_order):
        cfg_df = df[df["config"] == config_name]
        n_subjects = max((len(p) for p in cfg_df["psf_list"] if p), default=0)
        if n_subjects == 0:
            ax.set_visible(False)
            continue

        # First gather per-subject pass rates for each agent
        agent_psf: dict[str, np.ndarray] = {}
        agent_scalar: dict[str, tuple[float, float]] = {}
        for agent in keep:
            rows = cfg_df[cfg_df["agent"] == agent]
            if rows.empty:
                continue
            agent_scalar[agent] = (rows["mean_return"].mean(), rows["cvar_10"].mean())
            psf_arr = np.array(
                [p for p in rows["psf_list"] if len(p) == n_subjects]
            )
            agent_psf[agent] = psf_arr.mean(axis=0)

        if not agent_scalar:
            ax.set_visible(False)
            continue

        # Drop subjects that are universally infeasible (every agent fails > 99%).
        # Those axes collapse to 0 and add no information.
        psf_stack = np.stack(list(agent_psf.values()))
        infeasible = (psf_stack.min(axis=0) > 0.99)
        feasible_subjects = [i for i in range(n_subjects) if not infeasible[i]]

        # Build per-agent vector:
        #   [mean_return_normalized, cvar_normalized, pass_rate_S0, pass_rate_S1, ...]
        # Pass rates are kept as ABSOLUTE values in [0, 1] (no min-max
        # normalization). Mean Return and CVaR are divided by the
        # best agent's value on each axis, mapping the best to 1.0
        # and others to their proportional fraction. This preserves
        # the magnitude of relative differences instead of amplifying
        # tiny gaps to 0-vs-1 like full min-max normalization does.
        all_means = np.array([s[0] for s in agent_scalar.values()])
        all_cvars = np.array([s[1] for s in agent_scalar.values()])
        # Use the max-of-positive value as the scale, with a small floor.
        mean_scale = max(all_means.max(), 1e-6)
        cvar_scale = max(all_cvars.max(), 1e-6)

        agent_vectors: dict[str, np.ndarray] = {}
        for agent, (mean_r, cvar) in agent_scalar.items():
            pass_rates = 1.0 - agent_psf[agent][feasible_subjects]
            mean_n = max(0.0, mean_r / mean_scale)
            cvar_n = max(0.0, cvar / cvar_scale)
            agent_vectors[agent] = np.concatenate([[mean_n, cvar_n], pass_rates])

        normalized = agent_vectors  # already in [0, 1]

        labels = ["Mean\nReturn", "CVaR$_{10\\%}$"] + [
            f"S{i} pass" for i in feasible_subjects
        ]
        n_axes = len(labels)
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        # Plot in a stable order so SC-CVaR-DQN draws on top
        for agent in ["DQN", "CVaR-DQN", "SC-CVaR-DQN"]:
            if agent not in normalized:
                continue
            vec = normalized[agent]
            vec_closed = np.concatenate([vec, vec[:1]])
            color = AGENT_COLORS.get(agent, "#34495e")
            ax.plot(angles_closed, vec_closed,
                    color=color, linewidth=LINEWIDTHS[agent],
                    linestyle=LINESTYLES[agent], label=agent,
                    marker="o", markersize=6, markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=3 if agent == "SC-CVaR-DQN" else 2)
            ax.fill(angles_closed, vec_closed, color=color, alpha=0.12)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=11)
        ax.tick_params(axis="x", pad=18)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{config_name.capitalize()}", fontsize=14, fontweight="bold", pad=24)
        ax.grid(alpha=0.35)
        ax.spines["polar"].set_alpha(0.3)
        # Rotate so "Mean Return" is at the top
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

    handles = []
    labels_legend = []
    for agent in keep:
        if agent in df["agent"].unique():
            handles.append(
                plt.Line2D([0], [0], color=AGENT_COLORS[agent],
                           linewidth=LINEWIDTHS[agent], marker="o",
                           markersize=7, markeredgecolor="white",
                           markeredgewidth=1.0)
            )
            labels_legend.append(agent)
    fig.legend(
        handles, labels_legend,
        loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, -0.01),
        fontsize=12, frameon=True,
    )

    fig.suptitle(
        "RL Agent Profiles — larger area = better. "
        "Subject axes show absolute pass rate; aggregate axes scaled by best agent.",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "radar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved radar.png")


def plot_risk_return_pareto(results_dir: str, plots_dir: str) -> None:
    """Mean-return vs CVaR_10 scatter, RL agents only.

    Heuristics are intentionally omitted: they sit far from the RL
    Pareto frontier and squish the interesting region into a corner.
    Labels are positioned with hand-tuned offsets per agent so they
    do not overlap.
    """
    path = os.path.join(results_dir, "results_all.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    # Only the agents that matter for the risk-return discussion.
    # Value Iteration is intentionally omitted: it's a DP oracle (not a
    # competing learner), its number is already in the main results
    # table, and including it crowds the Small panel.
    keep = ["DQN", "CVaR-DQN", "SC-CVaR-DQN"]
    df = df[df["agent"].isin(keep)].copy()
    if df.empty:
        return

    # Per-agent label offset (dx, dy) in axis units; three clean directions.
    OFFSETS = {
        "DQN":         ( 0.20, -0.20),
        "CVaR-DQN":    (-0.20,  0.20),
        "SC-CVaR-DQN": ( 0.20,  0.20),
    }
    HA = {"DQN": "left", "CVaR-DQN": "right", "SC-CVaR-DQN": "left"}
    VA = {"DQN": "top", "CVaR-DQN": "bottom", "SC-CVaR-DQN": "bottom"}

    config_order = [c for c in ["small", "medium", "large"] if c in df["config"].unique()]
    fig, axes = plt.subplots(1, len(config_order), figsize=(5.5 * len(config_order), 5.2), squeeze=False)

    for ax, config_name in zip(axes[0], config_order):
        cfg_df = df[df["config"] == config_name]
        agg = cfg_df.groupby("agent").agg({
            "mean_return": ["mean", "std"], "cvar_10": ["mean", "std"]
        })

        all_x = list(agg[("mean_return", "mean")])
        all_y = list(agg[("cvar_10", "mean")])
        # Zoom to RL region with comfortable padding for labels
        pad_x = max(0.6, 0.25 * (max(all_x) - min(all_x) + 0.1))
        pad_y = max(0.6, 0.25 * (max(all_y) - min(all_y) + 0.1))
        x_lo, x_hi = min(all_x) - pad_x, max(all_x) + pad_x
        y_lo, y_hi = min(all_y) - pad_y, max(all_y) + pad_y

        ax.plot([min(x_lo, y_lo), max(x_hi, y_hi)], [min(x_lo, y_lo), max(x_hi, y_hi)],
                color="#cccccc", linestyle="--", linewidth=1, zorder=1)
        ax.text(0.97, 0.04, "risk-free line (CVaR = mean)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#888888", style="italic")

        for agent in keep:
            if agent not in agg.index:
                continue
            x = agg.loc[agent, ("mean_return", "mean")]
            y = agg.loc[agent, ("cvar_10", "mean")]
            xe = agg.loc[agent, ("mean_return", "std")] or 0
            ye = agg.loc[agent, ("cvar_10", "std")] or 0
            color = AGENT_COLORS.get(agent, "#34495e")
            ax.errorbar(x, y, xerr=xe, yerr=ye,
                        fmt="o", color=color, markersize=15,
                        markeredgecolor="black", markeredgewidth=1.3,
                        ecolor=color, elinewidth=1.5, capsize=4, zorder=3)
            dx, dy = OFFSETS.get(agent, (0.15, 0.15))
            ax.annotate(
                agent, (x, y),
                xytext=(x + dx, y + dy),
                fontsize=10, fontweight="bold",
                ha=HA.get(agent, "left"), va=VA.get(agent, "bottom"),
                color=color,
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.6, lw=0.8),
            )

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Mean Return", fontsize=11)
        ax.set_ylabel("CVaR$_{10\\%}$ (worst-case)", fontsize=11)
        ax.set_title(f"{config_name.capitalize()}", fontsize=13)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Risk-Return Frontier — RL agents (mean $\\pm$ std over 5 seeds)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "risk_return_pareto.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved risk_return_pareto.png")


def plot_learning_curves(results_dir: str, plots_dir: str) -> None:
    """Learning curves with variance bands across seeds."""
    for config_name in ["small", "medium", "large"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Learning Curves — {config_name.capitalize()} Config")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (smoothed)")

        for agent_name, color in [
            ("dqn", "#e74c3c"),
            ("cvar_dqn", "#9b59b6"),
            ("sc_cvar_dqn", "#16a085"),
        ]:
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
    agent_order = [a for a in ["DQN", "CVaR-DQN", "SC-CVaR-DQN"] if a in df["agent"].unique()]
    colors = [AGENT_COLORS.get(a, "#34495e") for a in agent_order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Ablation Study — Medium Config", fontsize=14)

    for ax, metric, title in zip(axes, metrics, titles):
        summary = df.groupby(["ablation", "agent"])[metric].mean().reset_index()
        pivot = summary.pivot(index="ablation", columns="agent", values=metric)
        pivot = pivot[agent_order]
        pivot.plot(kind="bar", ax=ax, color=colors)
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
    agent_order = [a for a in ["DQN", "CVaR-DQN", "SC-CVaR-DQN"] if a in df["agent"].unique()]
    colors = [AGENT_COLORS.get(a, "#34495e") for a in agent_order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Robustness Tests — RL agents under distribution shift", fontsize=14)

    for ax, metric, title in zip(axes, metrics, titles):
        summary = df.groupby(["test", "agent"])[metric].mean().reset_index()
        pivot = summary.pivot(index="test", columns="agent", values=metric)
        pivot = pivot[agent_order]
        pivot.plot(kind="bar", ax=ax, color=colors)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "robustness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved robustness.png")


def plot_per_subject_failure(results_dir: str, plots_dir: str) -> None:
    """Grouped bar chart of per-subject failure rates for DQN / CVaR-DQN / SC-CVaR-DQN.

    Headline visualization for Subject-Constrained CVaR-DQN: shows that the
    constrained agent eliminates the sacrificed-subject failure mode.
    """
    path = os.path.join(results_dir, "results_all.csv")
    if not os.path.exists(path):
        logger.warning("results_all.csv not found, skipping per-subject failure plot")
        return

    df = pd.read_csv(path)
    df = df[df["agent"].isin(["DQN", "CVaR-DQN", "SC-CVaR-DQN"])].copy()
    if df.empty:
        return

    def parse_psf(s: str) -> list[float]:
        if pd.isna(s) or s == "":
            return []
        return [float(x) for x in str(s).split(";")]

    df["psf_list"] = df["per_subject_failure_rate"].apply(parse_psf)

    configs = ["small", "medium", "large"]
    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 5), squeeze=False)
    agents_order = ["DQN", "CVaR-DQN", "SC-CVaR-DQN"]

    for ax, config_name in zip(axes[0], configs):
        sub = df[df["config"] == config_name]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Number of subjects from the first non-empty psf list
        n_subjects = max((len(p) for p in sub["psf_list"] if p), default=0)
        if n_subjects == 0:
            ax.set_visible(False)
            continue

        # Mean over seeds per agent per subject
        means = {a: np.zeros(n_subjects) for a in agents_order}
        stds = {a: np.zeros(n_subjects) for a in agents_order}
        for a in agents_order:
            agent_rows = sub[sub["agent"] == a]
            if agent_rows.empty:
                continue
            arr = np.array([p for p in agent_rows["psf_list"] if len(p) == n_subjects])
            if len(arr) == 0:
                continue
            means[a] = arr.mean(axis=0)
            stds[a] = arr.std(axis=0)

        x = np.arange(n_subjects)
        width = 0.25
        for i, a in enumerate(agents_order):
            offset = (i - 1) * width
            ax.bar(
                x + offset, means[a], width, yerr=stds[a],
                color=AGENT_COLORS.get(a, "#34495e"), label=a, capsize=3,
            )

        ax.set_title(f"{config_name.capitalize()}")
        ax.set_xlabel("Subject index")
        ax.set_ylabel("Failure rate (score < 0.5 * w_i)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i}" for i in range(n_subjects)])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-subject failure rate (mean over 5 seeds)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "per_subject_failure.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per_subject_failure.png")


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
    plot_radar(args.results_dir, plots_dir)
    plot_learning_curves(args.results_dir, plots_dir)
    plot_ablation(args.results_dir, plots_dir)
    plot_robustness(args.results_dir, plots_dir)
    plot_per_subject_failure(args.results_dir, plots_dir)

    logger.info("All plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
