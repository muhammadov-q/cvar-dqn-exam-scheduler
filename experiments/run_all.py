"""Run all experiments: train and evaluate all agents on all env sizes."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.study_env import EnvConfig, StudyEnv
from agents.heuristics import UniformAgent, MostUrgentFirstAgent, LowestKnowledgeFirstAgent
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.cvar_dqn import CVaRDQNAgent
from agents.subject_cvar_dqn import SubjectCVaRDQNAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]
EVAL_EPISODES = 1000


def bootstrap_ci(data: list[float], n_bootstrap: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    """Compute bootstrapped confidence interval."""
    arr = np.array(data)
    boot_means = np.array([
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def make_row(config_name: str, agent_name: str, seed: int, result: dict) -> dict:
    """Build a result row from an agent's evaluate() output.

    Per-subject failure rates are serialized as a semicolon-joined
    string so they fit cleanly in a CSV column.
    """
    ci_lo, ci_hi = bootstrap_ci(result["all_returns"])
    per_subj = result.get("per_subject_failure_rate", [])
    return {
        "config": config_name,
        "agent": agent_name,
        "seed": seed,
        "mean_return": result["mean_total_reward"],
        "std_return": result["std_total_reward"],
        "mean_worst_exam": result["mean_worst_exam"],
        "failure_rate": result["failure_rate"],
        "cvar_10": result["cvar_10"],
        "per_subject_failure_rate": ";".join(f"{x:.4f}" for x in per_subj),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def run_experiment(
    config_name: str,
    config: EnvConfig,
    seeds: list[int],
    n_train_episodes: int,
    n_eval_episodes: int,
    results_dir: str,
) -> pd.DataFrame:
    """Run all agents on a single environment configuration."""
    rows: list[dict] = []

    for seed in seeds:
        logger.info("=== Config: %s, Seed: %d ===", config_name, seed)

        env = StudyEnv(config)

        # --- Heuristic agents (no training needed) ---
        heuristics = {
            "Uniform": UniformAgent(env),
            "MostUrgentFirst": MostUrgentFirstAgent(env),
            "LowestKnowledgeFirst": LowestKnowledgeFirstAgent(env),
        }
        for name, agent in heuristics.items():
            logger.info("Evaluating %s...", name)
            result = agent.evaluate(n_episodes=n_eval_episodes, seed=seed)
            rows.append(make_row(config_name, name, seed, result))

        # --- Value Iteration (small only) ---
        if config_name == "small":
            logger.info("Training ValueIteration...")
            vi = ValueIterationAgent(env)
            vi.train()
            result = vi.evaluate(n_episodes=n_eval_episodes, seed=seed)
            rows.append(make_row(config_name, "ValueIteration", seed, result))

        # --- Q-Learning (small and medium) ---
        if config_name in ("small", "medium"):
            logger.info("Training Q-Learning...")
            ql = QLearningAgent(env)
            ql.train(n_episodes=n_train_episodes, seed=seed)
            result = ql.evaluate(n_episodes=n_eval_episodes, seed=seed)
            rows.append(make_row(config_name, "QLearning", seed, result))

        # --- DQN ---
        logger.info("Training DQN...")
        dqn = DQNAgent(env)
        learning_curve = dqn.train(n_episodes=n_train_episodes, seed=seed)
        result = dqn.evaluate(n_episodes=n_eval_episodes, seed=seed)
        rows.append(make_row(config_name, "DQN", seed, result))

        lc_path = os.path.join(results_dir, f"learning_curve_dqn_{config_name}_seed{seed}.csv")
        pd.DataFrame({"episode": range(len(learning_curve)), "return": learning_curve}).to_csv(lc_path, index=False)

        # --- CVaR-DQN ---
        logger.info("Training CVaR-DQN...")
        cvar_dqn = CVaRDQNAgent(env)
        learning_curve = cvar_dqn.train(n_episodes=n_train_episodes, seed=seed)
        result = cvar_dqn.evaluate(n_episodes=n_eval_episodes, seed=seed)
        rows.append(make_row(config_name, "CVaR-DQN", seed, result))

        lc_path = os.path.join(results_dir, f"learning_curve_cvar_dqn_{config_name}_seed{seed}.csv")
        pd.DataFrame({"episode": range(len(learning_curve)), "return": learning_curve}).to_csv(lc_path, index=False)

        # --- Subject-Constrained CVaR-DQN (main novel contribution) ---
        logger.info("Training Subject-Constrained CVaR-DQN...")
        sc_cvar_dqn = SubjectCVaRDQNAgent(env)
        learning_curve = sc_cvar_dqn.train(n_episodes=n_train_episodes, seed=seed)
        result = sc_cvar_dqn.evaluate(n_episodes=n_eval_episodes, seed=seed)
        rows.append(make_row(config_name, "SC-CVaR-DQN", seed, result))

        lc_path = os.path.join(results_dir, f"learning_curve_sc_cvar_dqn_{config_name}_seed{seed}.csv")
        pd.DataFrame({"episode": range(len(learning_curve)), "return": learning_curve}).to_csv(lc_path, index=False)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--configs", nargs="+", default=["small", "medium", "large"],
                        choices=["small", "medium", "large"])
    parser.add_argument("--n-train", type=int, default=5000, help="Training episodes")
    parser.add_argument("--n-eval", type=int, default=1000, help="Evaluation episodes")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)

    config_map = {
        "small": EnvConfig.small(),
        "medium": EnvConfig.medium(),
        "large": EnvConfig.large(),
    }

    all_dfs: list[pd.DataFrame] = []
    for config_name in args.configs:
        logger.info(">>> Starting experiments for config: %s", config_name)
        df = run_experiment(
            config_name=config_name,
            config=config_map[config_name],
            seeds=args.seeds,
            n_train_episodes=args.n_train,
            n_eval_episodes=args.n_eval,
            results_dir=args.results_dir,
        )
        all_dfs.append(df)
        df.to_csv(os.path.join(args.results_dir, f"results_{config_name}.csv"), index=False)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(args.results_dir, "results_all.csv"), index=False)

    # Print summary
    summary = combined.groupby(["config", "agent"]).agg({
        "mean_return": ["mean", "std"],
        "mean_worst_exam": "mean",
        "failure_rate": "mean",
        "cvar_10": "mean",
    }).round(4)
    print("\n=== RESULTS SUMMARY ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
