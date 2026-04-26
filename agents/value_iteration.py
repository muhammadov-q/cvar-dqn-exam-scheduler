"""Value Iteration agent (exact DP) for small environment configurations."""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import numpy as np

from environment.study_env import EnvConfig, StudyEnv

logger = logging.getLogger(__name__)


class ValueIterationAgent:
    """Exact dynamic programming via value iteration.

    Only feasible for the small environment configuration due to
    state space size. Serves as ground truth for validating other agents.
    """

    def __init__(
        self,
        env: StudyEnv,
        gamma: float = 0.99,
        theta: float = 1e-6,
        max_iterations: int = 1000,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        c = env.config
        self.n_subjects = c.n_subjects
        self.n_actions = c.n_subjects + 1

        # Enumerate all states
        knowledge_ranges = [range(c.max_knowledge + 1) for _ in range(c.n_subjects)]
        energy_range = range(1, c.max_energy + 1)
        day_range = range(1, c.n_days + 2)  # +1 for terminal check, +1 for range end

        self._states: list[tuple] = []
        self._state_index: dict[tuple, int] = {}
        for combo in product(*knowledge_ranges, energy_range, day_range):
            idx = len(self._states)
            self._states.append(combo)
            self._state_index[combo] = idx

        n_states = len(self._states)
        logger.info("Value iteration: %d states, %d actions", n_states, self.n_actions)

        self.V = np.zeros(n_states, dtype=np.float64)
        self.policy = np.zeros(n_states, dtype=np.int64)

    def _simulate_transition(
        self, state: tuple, action: int, noise: int
    ) -> tuple[tuple, float]:
        """Simulate a single transition given state, action, and energy noise.

        Returns (next_state_tuple, reward).
        """
        c = self.env.config
        knowledge = list(state[: self.n_subjects])
        energy = state[self.n_subjects]
        day = state[self.n_subjects + 1]

        is_rest = action == self.n_subjects
        studied = None if is_rest else action

        # Forgetting
        for i in range(self.n_subjects):
            if i == studied:
                continue
            lam = c.forgetting_rates[i]
            knowledge[i] = max(0, int(np.floor(knowledge[i] * np.exp(-lam))))

        # Learning
        if studied is not None:
            gain = (energy / c.max_energy) * c.max_gain
            prereqs = c.prerequisites.get(studied, [])
            if prereqs:
                min_ratio = min(knowledge[p] / c.max_knowledge for p in prereqs)
                bonus = c.prerequisite_alpha * min_ratio if min_ratio >= c.prerequisite_beta else 0.0
            else:
                bonus = 0.0
            gain = gain * (1.0 + bonus)
            knowledge[studied] = min(c.max_knowledge, knowledge[studied] + int(np.floor(gain)))

        # Reward
        reward = c.step_cost
        exam_day_subjects = {}
        for subj, d in c.exam_schedule.items():
            exam_day_subjects.setdefault(d, []).append(subj)
        if day in exam_day_subjects:
            for subj in exam_day_subjects[day]:
                score = c.exam_weights[subj] / (
                    1.0 + np.exp(-(knowledge[subj] - c.max_knowledge / 2.0))
                )
                reward += score

        # Energy update
        study_cost = 0 if is_rest else 1
        extra_recovery = 1 if is_rest else 0
        new_energy = int(np.clip(
            energy - study_cost + 1 + extra_recovery + noise,
            1, c.max_energy,
        ))

        new_day = day + 1
        next_state = (*knowledge, new_energy, new_day)
        return next_state, reward

    def train(self) -> None:
        """Run value iteration until convergence."""
        c = self.env.config
        noise_vals = c.energy_noise_range
        noise_prob = 1.0 / len(noise_vals)

        for iteration in range(self.max_iterations):
            delta = 0.0
            for s_idx, state in enumerate(self._states):
                day = state[self.n_subjects + 1]
                # Terminal states: day > n_days
                if day > c.n_days:
                    continue

                best_value = -np.inf
                best_action = 0
                for a in range(self.n_actions):
                    q_value = 0.0
                    for noise in noise_vals:
                        ns, r = self._simulate_transition(state, a, noise)
                        ns_idx = self._state_index.get(ns)
                        if ns_idx is not None:
                            q_value += noise_prob * (r + self.gamma * self.V[ns_idx])
                        else:
                            q_value += noise_prob * r
                    if q_value > best_value:
                        best_value = q_value
                        best_action = a

                delta = max(delta, abs(best_value - self.V[s_idx]))
                self.V[s_idx] = best_value
                self.policy[s_idx] = best_action

            if iteration % 50 == 0:
                logger.info("VI iteration %d, delta=%.6f", iteration, delta)
            if delta < self.theta:
                logger.info("VI converged after %d iterations (delta=%.8f)", iteration, delta)
                break

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using the learned policy."""
        state = tuple(int(x) for x in obs)
        s_idx = self._state_index.get(state)
        if s_idx is None:
            logger.warning("Unknown state %s, returning rest action", state)
            return self.n_subjects
        return int(self.policy[s_idx])

    def evaluate(self, n_episodes: int = 1000, seed: int = 42) -> dict:
        """Evaluate the agent over multiple episodes."""
        all_returns: list[float] = []
        all_worst_exam: list[float] = []
        rng = np.random.default_rng(seed)

        for ep in range(n_episodes):
            ep_seed = int(rng.integers(0, 2**31))
            obs, info = self.env.reset(seed=ep_seed)
            total_reward = 0.0
            exam_scores: dict[int, float] = {}
            done = False

            while not done:
                action = self.select_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if "exam_scores" in info:
                    exam_scores.update(info["exam_scores"])
                done = terminated or truncated

            all_returns.append(total_reward)
            if exam_scores:
                all_worst_exam.append(min(exam_scores.values()))
            else:
                all_worst_exam.append(0.0)

        returns_arr = np.array(all_returns)
        worst_arr = np.array(all_worst_exam)
        cutoff = int(np.ceil(len(returns_arr) * 0.1))
        sorted_returns = np.sort(returns_arr)
        cvar_10 = float(np.mean(sorted_returns[:cutoff]))

        return {
            "mean_total_reward": float(np.mean(returns_arr)),
            "std_total_reward": float(np.std(returns_arr)),
            "mean_worst_exam": float(np.mean(worst_arr)),
            "failure_rate": float(np.mean(
                worst_arr < min(self.env.config.exam_weights) / (
                    1.0 + np.exp(-(1 - self.env.config.max_knowledge / 2.0))
                )
            )),
            "cvar_10": cvar_10,
            "all_returns": all_returns,
        }
