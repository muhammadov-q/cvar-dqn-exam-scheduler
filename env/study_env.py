"""Gymnasium environment for exam study scheduling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration for the study environment."""

    n_subjects: int = 3
    max_knowledge: int = 5
    n_days: int = 7
    max_energy: int = 5
    max_gain: int = 2
    forgetting_rates: list[float] | None = None
    prerequisites: dict[int, list[int]] | None = None
    exam_schedule: dict[int, int] | None = None
    exam_weights: list[float] | None = None
    prerequisite_alpha: float = 0.3
    prerequisite_beta: float = 0.3
    step_cost: float = -0.1
    energy_noise_range: list[int] = field(default_factory=lambda: [-1, 0, 1])

    @staticmethod
    def small() -> EnvConfig:
        """Small configuration: N=3, K=5, D=7."""
        return EnvConfig(
            n_subjects=3,
            max_knowledge=5,
            n_days=7,
            max_gain=2,
            forgetting_rates=[0.05, 0.15, 0.1],
            prerequisites={1: [0]},  # Physics requires Math
            exam_schedule={0: 4, 1: 6, 2: 5},  # Math day4, Physics day6, History day5
            exam_weights=[3.0, 4.0, 2.0],
        )

    @staticmethod
    def medium() -> EnvConfig:
        """Medium configuration: N=4, K=8, D=10."""
        return EnvConfig(
            n_subjects=4,
            max_knowledge=8,
            n_days=10,
            max_gain=2,
            forgetting_rates=[0.05, 0.15, 0.1, 0.08],
            prerequisites={1: [0]},  # Physics requires Math
            exam_schedule={0: 4, 1: 7, 2: 6, 3: 9},
            exam_weights=[3.0, 4.0, 2.0, 3.0],
        )

    @staticmethod
    def large() -> EnvConfig:
        """Large configuration: N=5, K=10, D=14."""
        return EnvConfig(
            n_subjects=5,
            max_knowledge=10,
            n_days=14,
            max_gain=3,
            forgetting_rates=[0.05, 0.15, 0.1, 0.08, 0.12],
            prerequisites={1: [0], 2: [1]},  # Math->Physics->Engineering
            exam_schedule={0: 5, 1: 8, 2: 11, 3: 13, 4: 14},
            exam_weights=[3.0, 4.0, 5.0, 2.0, 3.0],
        )


class StudyEnv(gym.Env):
    """Exam study scheduling environment.

    The agent decides which subject to study each day, considering
    forgetting, energy, and subject dependencies. Rewards are earned
    on exam days based on knowledge level.

    State: (knowledge_vector, energy, current_day)
    Actions: study subject 0..N-1, or rest (action N)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig.small()
        self.render_mode = render_mode
        c = self.config

        # Defaults for optional config fields
        if c.forgetting_rates is None:
            c.forgetting_rates = [0.1] * c.n_subjects
        if c.prerequisites is None:
            c.prerequisites = {}
        if c.exam_schedule is None:
            # Spread exams evenly
            days = np.linspace(c.n_days // 2, c.n_days, c.n_subjects, dtype=int)
            c.exam_schedule = {i: int(d) for i, d in enumerate(days)}
        if c.exam_weights is None:
            c.exam_weights = [1.0] * c.n_subjects

        # Observation space: knowledge per subject + energy + current_day
        # knowledge[i] in [0, K], energy in [1, E], day in [1, D]
        obs_highs = np.array(
            [c.max_knowledge] * c.n_subjects + [c.max_energy, c.n_days],
            dtype=np.int64,
        )
        obs_lows = np.array(
            [0] * c.n_subjects + [1, 1],
            dtype=np.int64,
        )
        self.observation_space = spaces.Box(
            low=obs_lows, high=obs_highs, dtype=np.int64
        )

        # Action space: study subject 0..N-1, or rest (action N)
        self.action_space = spaces.Discrete(c.n_subjects + 1)

        # Internal state
        self._knowledge: np.ndarray = np.zeros(c.n_subjects, dtype=np.int64)
        self._energy: int = c.max_energy
        self._day: int = 1

        # Pre-compute which days have exams and which subjects
        self._exam_day_subjects: dict[int, list[int]] = {}
        for subj, day in c.exam_schedule.items():
            self._exam_day_subjects.setdefault(day, []).append(subj)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self._knowledge,
            np.array([self._energy, self._day], dtype=np.int64),
        ])

    def _get_info(self) -> dict[str, Any]:
        return {
            "knowledge": self._knowledge.copy(),
            "energy": self._energy,
            "day": self._day,
            "exam_schedule": self.config.exam_schedule,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the start of the study period."""
        super().reset(seed=seed)
        self._knowledge = np.zeros(self.config.n_subjects, dtype=np.int64)
        self._energy = self.config.max_energy
        self._day = 1
        return self._get_obs(), self._get_info()

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _prerequisite_bonus(self, action: int) -> float:
        """Compute prerequisite bonus for studying a subject."""
        c = self.config
        prereqs = c.prerequisites.get(action, [])
        if not prereqs:
            return 0.0
        min_ratio = min(self._knowledge[p] / c.max_knowledge for p in prereqs)
        if min_ratio < c.prerequisite_beta:
            return 0.0
        return c.prerequisite_alpha * min_ratio

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one day of the study schedule."""
        c = self.config
        assert self.action_space.contains(action), f"Invalid action {action}"

        is_rest = action == c.n_subjects
        studied_subject = None if is_rest else action

        # --- Apply forgetting to all subjects NOT studied ---
        for i in range(c.n_subjects):
            if i == studied_subject:
                continue
            lam = c.forgetting_rates[i]
            self._knowledge[i] = max(
                0, int(np.floor(self._knowledge[i] * np.exp(-lam)))
            )

        # --- Apply learning for studied subject ---
        if studied_subject is not None:
            gain = (self._energy / c.max_energy) * c.max_gain
            bonus = self._prerequisite_bonus(studied_subject)
            gain = gain * (1.0 + bonus)
            self._knowledge[studied_subject] = min(
                c.max_knowledge,
                self._knowledge[studied_subject] + int(np.floor(gain)),
            )

        # --- Compute reward ---
        reward = c.step_cost

        # Exam rewards for subjects tested today
        exam_scores: dict[int, float] = {}
        if self._day in self._exam_day_subjects:
            for subj in self._exam_day_subjects[self._day]:
                score = c.exam_weights[subj] * self._sigmoid(
                    self._knowledge[subj] - c.max_knowledge / 2.0
                )
                exam_scores[subj] = score
                reward += score

        # --- Update energy ---
        study_cost = 0 if is_rest else 1
        extra_recovery = 1 if is_rest else 0
        noise = self.np_random.choice(c.energy_noise_range)
        self._energy = int(np.clip(
            self._energy - study_cost + 1 + extra_recovery + noise,
            1,
            c.max_energy,
        ))

        # --- Advance day ---
        self._day += 1
        terminated = self._day > c.n_days
        truncated = False

        info = self._get_info()
        info["exam_scores"] = exam_scores

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        """Render current state to console."""
        c = self.config
        subjects = [f"S{i}:{self._knowledge[i]}/{c.max_knowledge}" for i in range(c.n_subjects)]
        print(
            f"Day {self._day}/{c.n_days} | "
            f"Energy {self._energy}/{c.max_energy} | "
            f"Knowledge: {', '.join(subjects)}"
        )

    def state_to_tuple(self) -> tuple:
        """Convert current state to a hashable tuple (for tabular methods)."""
        return (*self._knowledge.tolist(), self._energy, self._day)

    def get_state_size(self) -> int:
        """Return the dimensionality of the flat observation vector."""
        return self.config.n_subjects + 2
