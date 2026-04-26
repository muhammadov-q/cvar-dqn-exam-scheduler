"""Tabular Q-learning agent for the study scheduling environment."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from tqdm import trange

from environment.study_env import StudyEnv

logger = logging.getLogger(__name__)


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    Works for small and medium environment configurations.
    """

    def __init__(
        self,
        env: StudyEnv,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_actions = env.config.n_subjects + 1

        self.Q: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def _state_key(self, obs: np.ndarray) -> tuple:
        return tuple(int(x) for x in obs)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        state = self._state_key(obs)
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def train(
        self,
        n_episodes: int = 10000,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[float]:
        """Train the agent using Q-learning.

        Returns list of episode returns (learning curve).
        """
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        episode_returns: list[float] = []
        self.epsilon = self.epsilon_start

        iterator = trange(n_episodes, desc="Q-learning") if verbose else range(n_episodes)
        for ep in iterator:
            ep_seed = int(rng.integers(0, 2**31))
            obs, _ = self.env.reset(seed=ep_seed)
            total_reward = 0.0
            done = False

            while not done:
                state = self._state_key(obs)
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Q-learning update
                next_state = self._state_key(next_obs)
                best_next = np.max(self.Q[next_state]) if not done else 0.0
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error

                obs = next_obs

            episode_returns.append(total_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if verbose and ep % 1000 == 0:
                recent = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
                logger.info("Episode %d, avg return: %.3f, epsilon: %.3f", ep, recent, self.epsilon)

        return episode_returns

    def evaluate(self, n_episodes: int = 1000, seed: int = 42) -> dict:
        """Evaluate the agent over multiple episodes (greedy policy)."""
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
                action = self.select_action(obs, greedy=True)
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
