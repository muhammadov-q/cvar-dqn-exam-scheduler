"""CVaR-DQN agent: distributional RL with risk-sensitive optimization.

Uses quantile regression (Dabney et al., 2018) to learn the full return
distribution, then optimizes CVaR (Conditional Value at Risk) to focus
on worst-case outcomes.
"""

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


class QuantileNetwork(nn.Module):
    """Network that outputs N_quantiles values per action.

    For each action, the network outputs a set of quantile values
    that together approximate the full return distribution.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_quantiles: int = 32,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions * n_quantiles))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns tensor of shape (batch_size, n_actions, n_quantiles).
        """
        out = self.network(x)
        return out.view(-1, self.n_actions, self.n_quantiles)


class CVaRDQNAgent:
    """CVaR-DQN: Distributional RL with risk-sensitive action selection.

    Instead of optimizing expected return, this agent optimizes CVaR_alpha,
    which is the mean of the worst alpha-fraction of return quantiles.
    This encourages policies that avoid catastrophic outcomes (e.g., failing
    any single exam).
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
        n_quantiles: int = 32,
        cvar_alpha: float = 0.1,
        hidden_sizes: tuple[int, ...] = (128, 128),
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
        self.n_quantiles = n_quantiles
        self.cvar_alpha = cvar_alpha
        self.n_actions = env.config.n_subjects + 1

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        state_dim = env.get_state_size()
        self.online_net = QuantileNetwork(
            state_dim, self.n_actions, n_quantiles, hidden_sizes
        ).to(self.device)
        self.target_net = QuantileNetwork(
            state_dim, self.n_actions, n_quantiles, hidden_sizes
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        # Quantile midpoints: tau_hat_i = (tau_i + tau_{i+1}) / 2
        taus = torch.linspace(0, 1, n_quantiles + 1, device=self.device)
        self.tau_hat = ((taus[:-1] + taus[1:]) / 2).unsqueeze(0)  # (1, n_quantiles)

        # Number of quantiles in the CVaR tail
        self.cvar_n = max(1, int(n_quantiles * cvar_alpha))

        # Normalization
        c = env.config
        self._obs_low = np.array([0] * c.n_subjects + [1, 1], dtype=np.float32)
        self._obs_high = np.array(
            [c.max_knowledge] * c.n_subjects + [c.max_energy, c.n_days],
            dtype=np.float32,
        )
        self._obs_range = np.maximum(self._obs_high - self._obs_low, 1e-8)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32) - self._obs_low) / self._obs_range

    def _cvar_value(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Compute CVaR from quantile values.

        Args:
            quantiles: shape (batch_size, n_actions, n_quantiles)

        Returns:
            CVaR values of shape (batch_size, n_actions)
        """
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        return sorted_q[..., : self.cvar_n].mean(dim=-1)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Select action using CVaR-based epsilon-greedy policy."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.tensor(
                self._normalize(obs), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            quantiles = self.online_net(state_t)  # (1, n_actions, n_quantiles)
            cvar_values = self._cvar_value(quantiles)  # (1, n_actions)
            return int(cvar_values.argmax(dim=1).item())

    def _quantile_huber_loss(
        self,
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """Compute quantile regression loss with Huber penalty.

        Args:
            quantiles: predicted quantile values, shape (batch, n_quantiles)
            target_quantiles: target quantile values, shape (batch, n_quantiles)
            kappa: Huber loss threshold

        Returns:
            Scalar loss.
        """
        batch_size = quantiles.shape[0]
        n_q = self.n_quantiles

        # Expand for pairwise comparison: (batch, n_q, 1) vs (batch, 1, n_q)
        pred = quantiles.unsqueeze(2)       # (batch, n_q, 1)
        target = target_quantiles.unsqueeze(1)  # (batch, 1, n_q)

        # Pairwise TD errors
        td_error = target - pred  # (batch, n_q, n_q)

        # Huber loss element
        huber = torch.where(
            td_error.abs() <= kappa,
            0.5 * td_error.pow(2),
            kappa * (td_error.abs() - 0.5 * kappa),
        )

        # Quantile weights: tau_hat for each predicted quantile
        tau_hat = self.tau_hat.unsqueeze(2)  # (1, n_q, 1)
        weight = torch.abs(tau_hat - (td_error < 0).float())  # (batch, n_q, n_q)

        loss = (weight * huber).sum(dim=2).mean(dim=1)  # (batch,)
        return loss.mean()

    def _update(self) -> float:
        """Perform one gradient step on a batch."""
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

        # Current quantile values for chosen actions: (batch, n_quantiles)
        all_quantiles = self.online_net(states)  # (batch, n_actions, n_quantiles)
        action_idx = actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_quantiles)
        current_quantiles = all_quantiles.gather(1, action_idx).squeeze(1)

        # Target quantile values
        with torch.no_grad():
            next_quantiles = self.target_net(next_states)  # (batch, n_actions, n_quantiles)

            # Use CVaR to select best next action (online net)
            online_next = self.online_net(next_states)
            next_cvar = self._cvar_value(online_next)
            best_actions = next_cvar.argmax(dim=1)  # (batch,)

            best_idx = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_quantiles)
            next_best_quantiles = next_quantiles.gather(1, best_idx).squeeze(1)

            # Bellman target for each quantile
            target_quantiles = rewards.unsqueeze(1) + self.gamma * next_best_quantiles * (
                1 - dones.unsqueeze(1)
            )

        loss = self._quantile_huber_loss(current_quantiles, target_quantiles)
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
        """Train the CVaR-DQN agent.

        Returns list of episode returns.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        episode_returns: list[float] = []
        total_steps = 0
        self.epsilon = self.epsilon_start

        iterator = trange(n_episodes, desc="CVaR-DQN") if verbose else range(n_episodes)
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

                if total_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                total_reward += reward
                obs = next_obs

            episode_returns.append(total_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if verbose and ep % 500 == 0:
                recent = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
                logger.info("CVaR-DQN ep %d, avg return: %.3f, epsilon: %.3f", ep, recent, self.epsilon)

        return episode_returns

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
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.online_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.online_net.state_dict())
