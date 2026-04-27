"""Heuristic baseline agents for the study scheduling environment."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from environment.study_env import StudyEnv

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, env: StudyEnv) -> None:
        self.env = env
        self.n_subjects = env.config.n_subjects

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """Select an action given the current observation."""
        ...

    def evaluate(self, n_episodes: int = 1000, seed: int = 42) -> dict:
        """Evaluate the agent over multiple episodes.

        Returns dict with mean_total_reward, worst_exam_scores, failure_rate,
        cvar_10, all_returns.
        """
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
        alpha = 0.1
        cutoff = int(np.ceil(len(returns_arr) * alpha))
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


class UniformAgent(BaseAgent):
    """Round-robin through subjects."""

    def __init__(self, env: StudyEnv) -> None:
        super().__init__(env)
        self._counter = 0

    def select_action(self, obs: np.ndarray) -> int:
        action = self._counter % self.n_subjects
        self._counter += 1
        return action


class MostUrgentFirstAgent(BaseAgent):
    """Study the subject whose exam is soonest."""

    def select_action(self, obs: np.ndarray) -> int:
        current_day = int(obs[-1])
        schedule = self.env.config.exam_schedule
        # Filter to subjects whose exams haven't passed yet
        future = {
            subj: day for subj, day in schedule.items() if day >= current_day
        }
        if not future:
            return self.n_subjects  # rest
        # Pick subject with earliest upcoming exam
        return min(future, key=future.get)


class LowestKnowledgeFirstAgent(BaseAgent):
    """Study the subject with the lowest current knowledge."""

    def select_action(self, obs: np.ndarray) -> int:
        knowledge = obs[: self.n_subjects]
        return int(np.argmin(knowledge))
