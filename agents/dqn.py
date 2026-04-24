"""DQN and Double DQN agent for the study scheduling environment."""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from environment.study_env import StudyEnv

logger = logging.getLogger(__name__)


class Transition(NamedTuple):
    """A single experience tuple."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 50000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Multi-layer Q-network."""

    def __init__(self, state_dim: int, n_actions: int, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    """DQN with Double DQN variant, experience replay, and target network.

    Works for all environment configurations.
    """

    def __init__(
        self,
        env: StudyEnv,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_sizes: tuple[int, ...] = (128, 128),
        double_dqn: bool = True,
        device: str | None = None,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.n_actions = env.config.n_subjects + 1

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        state_dim = env.get_state_size()
        self.online_net = QNetwork(state_dim, self.n_actions, hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, self.n_actions, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        # Normalization parameters (computed from env config)
        c = env.config
        self._obs_low = np.array([0] * c.n_subjects + [1, 1], dtype=np.float32)
        self._obs_high = np.array(
            [c.max_knowledge] * c.n_subjects + [c.max_energy, c.n_days],
            dtype=np.float32,
        )
        self._obs_range = np.maximum(self._obs_high - self._obs_low, 1e-8)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [0, 1] range."""
        return (obs.astype(np.float32) - self._obs_low) / self._obs_range

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.tensor(
                self._normalize(obs), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def _update(self) -> float:
        """Perform one gradient step on a batch from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(
            np.array([self._normalize(t.state) for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device,
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device,
        )
        next_states = torch.tensor(
            np.array([self._normalize(t.next_state) for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device,
        )

        # Current Q-values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: online net selects, target net evaluates
                best_actions = self.online_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values

            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        n_episodes: int = 5000,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[float]:
        """Train the DQN agent.

        Returns list of episode returns (learning curve).
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        episode_returns: list[float] = []
        total_steps = 0
        self.epsilon = self.epsilon_start

        iterator = trange(n_episodes, desc="DQN") if verbose else range(n_episodes)
        for ep in iterator:
            ep_seed = int(rng.integers(0, 2**31))
            obs, _ = self.env.reset(seed=ep_seed)
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.push(Transition(obs, action, reward, next_obs, done))
                self._update()
                total_steps += 1

                # Update target network
                if total_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                total_reward += reward
                obs = next_obs

            episode_returns.append(total_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if verbose and ep % 500 == 0:
                recent = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
                logger.info("DQN ep %d, avg return: %.3f, epsilon: %.3f", ep, recent, self.epsilon)

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

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        self.online_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.online_net.state_dict())
