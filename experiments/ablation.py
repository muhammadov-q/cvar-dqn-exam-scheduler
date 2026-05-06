"""Ablation studies on the medium environment configuration."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.study_env import EnvConfig, StudyEnv
from agents.dqn import DQNAgent
from agents.cvar_dqn import CVaRDQNAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]


def make_ablation_configs() -> dict[str, EnvConfig]:
    """Create ablation variants of the medium config."""
    base = EnvConfig.medium()

    # 1. No forgetting
    no_forget = copy.deepcopy(base)
    no_forget.forgetting_rates = [0.0] * base.n_subjects

    # 2. Constant energy (max)
    const_energy = copy.deepcopy(base)
    const_energy.energy_noise_range = [0]

    # 3. No prerequisites
    no_prereq = copy.deepcopy(base)
    no_prereq.prerequisites = {}

    # 4. Equal exam weights
    equal_weights = copy.deepcopy(base)
    equal_weights.exam_weights = [1.0] * base.n_subjects

    # 5. Varied forgetting rates
    configs = {
        "baseline": base,
        "no_forgetting": no_forget,
        "constant_energy": const_energy,
        "no_prerequisites": no_prereq,
        "equal_weights": equal_weights,
    }

    for lam in [0.05, 0.1, 0.2, 0.4]:
        cfg = copy.deepcopy(base)
        cfg.forgetting_rates = [lam] * base.n_subjects
        configs[f"forget_rate_{lam}"] = cfg

    return configs


def run_ablation(
    n_train: int,
    n_eval: int,
    results_dir: str,
    seeds: list[int],
) -> pd.DataFrame:
    """Run ablation studies."""
    configs = make_ablation_configs()
    rows: list[dict] = []

    for ablation_name, config in configs.items():
        for seed in seeds:
            logger.info("Ablation: %s, seed: %d", ablation_name, seed)
            env = StudyEnv(config)

            # DQN
            dqn = DQNAgent(env)
            dqn.train(n_episodes=n_train, seed=seed, verbose=False)
            result = dqn.evaluate(n_episodes=n_eval, seed=seed)
            rows.append({
                "ablation": ablation_name, "agent": "DQN", "seed": seed,
                **{k: v for k, v in result.items() if k != "all_returns"},
            })

            # CVaR-DQN
            cvar = CVaRDQNAgent(env)
            cvar.train(n_episodes=n_train, seed=seed, verbose=False)
            result = cvar.evaluate(n_episodes=n_eval, seed=seed)
            rows.append({
                "ablation": ablation_name, "agent": "CVaR-DQN", "seed": seed,
                **{k: v for k, v in result.items() if k != "all_returns"},
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    df = run_ablation(args.n_train, args.n_eval, args.results_dir, args.seeds)
    df.to_csv(os.path.join(args.results_dir, "ablation_results.csv"), index=False)

    summary = df.groupby(["ablation", "agent"]).agg({
        "mean_total_reward": ["mean", "std"],
        "mean_worst_exam": "mean",
        "failure_rate": "mean",
        "cvar_10": "mean",
    }).round(4)
    print("\n=== ABLATION RESULTS ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
