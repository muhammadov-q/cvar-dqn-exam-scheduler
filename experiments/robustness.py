"""Robustness tests: train under normal conditions, test under perturbations."""

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
from agents.subject_cvar_dqn import SubjectCVaRDQNAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]


def run_robustness(
    n_train: int,
    n_eval: int,
    results_dir: str,
    seeds: list[int],
) -> pd.DataFrame:
    """Run robustness tests."""
    rows: list[dict] = []
    base_config = EnvConfig.medium()

    for seed in seeds:
        logger.info("=== Robustness tests, seed: %d ===", seed)

        # --- Train agents on normal conditions ---
        train_env = StudyEnv(base_config)

        dqn = DQNAgent(train_env)
        dqn.train(n_episodes=n_train, seed=seed, verbose=False)

        cvar = CVaRDQNAgent(train_env)
        cvar.train(n_episodes=n_train, seed=seed, verbose=False)

        sc = SubjectCVaRDQNAgent(train_env)
        sc.train(n_episodes=n_train, seed=seed, verbose=False)

        trained = [("DQN", dqn), ("CVaR-DQN", cvar), ("SC-CVaR-DQN", sc)]

        # --- Test 1: Wider energy noise ---
        wide_noise_config = copy.deepcopy(base_config)
        wide_noise_config.energy_noise_range = [-2, -1, 0, 1, 2]
        test_env = StudyEnv(wide_noise_config)

        for name, agent in trained:
            agent.env = test_env
            result = agent.evaluate(n_episodes=n_eval, seed=seed)
            rows.append({
                "test": "wide_energy_noise", "agent": name, "seed": seed,
                **{k: v for k, v in result.items() if k != "all_returns"},
            })
            agent.env = train_env  # restore

        # --- Test 2: Compressed exam schedule ---
        compressed_config = copy.deepcopy(base_config)
        # Move all exams closer together (last 4 days)
        n = compressed_config.n_subjects
        compressed_config.exam_schedule = {
            i: compressed_config.n_days - n + 1 + i for i in range(n)
        }
        test_env = StudyEnv(compressed_config)

        for name, agent in trained:
            agent.env = test_env
            result = agent.evaluate(n_episodes=n_eval, seed=seed)
            rows.append({
                "test": "compressed_exams", "agent": name, "seed": seed,
                **{k: v for k, v in result.items() if k != "all_returns"},
            })
            agent.env = train_env

        # --- Test 3: High variance (increased forgetting) ---
        high_var_config = copy.deepcopy(base_config)
        high_var_config.forgetting_rates = [r * 2.0 for r in base_config.forgetting_rates]
        high_var_config.energy_noise_range = [-2, -1, 0, 1, 2]
        test_env = StudyEnv(high_var_config)

        for name, agent in trained:
            agent.env = test_env
            result = agent.evaluate(n_episodes=n_eval, seed=seed)
            rows.append({
                "test": "high_variance", "agent": name, "seed": seed,
                **{k: v for k, v in result.items() if k != "all_returns"},
            })
            agent.env = train_env

        # --- Control: Normal conditions ---
        for name, agent in trained:
            result = agent.evaluate(n_episodes=n_eval, seed=seed)
            rows.append({
                "test": "normal", "agent": name, "seed": seed,
                **{k: v for k, v in result.items() if k != "all_returns"},
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness tests")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    df = run_robustness(args.n_train, args.n_eval, args.results_dir, args.seeds)
    df.to_csv(os.path.join(args.results_dir, "robustness_results.csv"), index=False)

    summary = df.groupby(["test", "agent"]).agg({
        "mean_total_reward": ["mean", "std"],
        "mean_worst_exam": "mean",
        "failure_rate": "mean",
        "cvar_10": "mean",
    }).round(4)
    print("\n=== ROBUSTNESS RESULTS ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
