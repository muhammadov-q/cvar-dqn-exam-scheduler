"""Subject-Constrained CVaR-DQN: distributional constrained RL.

Extends CVaR-DQN by learning a separate return distribution for EACH
subject (instead of a single distribution over the total return). At
action-selection time, the agent maximizes the expected total return
subject to per-subject CVaR constraints: the CVaR of every individual
subject's return must stay above a passing threshold T_i.

This is a Multi-Objective Distributional Constrained MDP. We use a
Lagrangian relaxation of the per-subject constraints:

    score(s, a) = sum_i E[Z_i(s, a)]
                  - lambda * sum_i max(0, T_i - CVaR_alpha(Z_i(s, a)))

with action selection a* = argmax_a score(s, a). For sufficiently
large lambda, the Lagrangian penalty dominates the score when any
constraint is violated, recovering the hard-constraint solution where
feasible and gracefully degrading to "minimize total violation" where
no feasible action exists. Crucially, the score is continuous in the
quantile predictions, so Bellman bootstrapping with the same rule
yields stable training (unlike a hard feasible/maximin switch).
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


class SubjectTransition(NamedTuple):
    state: np.ndarray
    action: int
    subject_rewards: np.ndarray  # shape (n_subjects,) per-subject reward at this step
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50000) -> None:
        self.buffer: deque[SubjectTransition] = deque(maxlen=capacity)

    def push(self, transition: SubjectTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[SubjectTransition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class SubjectQuantileNetwork(nn.Module):
    """Outputs n_subjects * n_quantiles values per action.

    Final layer shape: (n_actions, n_subjects, n_quantiles). Each
    (action, subject) slice approximates the quantile distribution of
    the return attributable to that subject from state s after taking
    that action.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_subjects: int,
        n_quantiles: int = 32,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_subjects = n_subjects
        self.n_quantiles = n_quantiles

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions * n_subjects * n_quantiles))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        return out.view(-1, self.n_actions, self.n_subjects, self.n_quantiles)


class SubjectCVaRDQNAgent:
    """Subject-Constrained CVaR-DQN.

    Lagrangian-relaxed objective at every step:
        a* = argmax_a  sum_i E[Z_i(s,a)]
                       - lambda * sum_i max(0, T_i - CVaR_alpha(Z_i(s,a)))

    Threshold rule (threshold_mode='half_weight'): T_i = 0.5 * w_i,
    the sigmoid-midpoint reward (reward earned by knowing K/2 of
    subject i). The Lagrangian multiplier ``lagrangian_lambda`` is
    chosen so that constraint violations dominate the expected-return
    term: with exam rewards O(w_i), lambda=10 makes a unit violation
    10x more costly than a unit of return.
    """

    def __init__(
        self,
        env: StudyEnv,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        n_quantiles: int = 32,
        cvar_alpha: float = 0.1,
        threshold_mode: str = "half_weight",
        threshold_scale: float = 0.5,
        lagrangian_lambda: float = 5.0,
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
        self.n_subjects = env.config.n_subjects

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        state_dim = env.get_state_size()
        self.online_net = SubjectQuantileNetwork(
            state_dim, self.n_actions, self.n_subjects, n_quantiles, hidden_sizes
        ).to(self.device)
        self.target_net = SubjectQuantileNetwork(
            state_dim, self.n_actions, self.n_subjects, n_quantiles, hidden_sizes
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        taus = torch.linspace(0, 1, n_quantiles + 1, device=self.device)
        self.tau_hat = ((taus[:-1] + taus[1:]) / 2).unsqueeze(0)  # (1, n_quantiles)

        self.cvar_n = max(1, int(n_quantiles * cvar_alpha))
        self.lagrangian_lambda = lagrangian_lambda

        weights = torch.tensor(env.config.exam_weights, dtype=torch.float32, device=self.device)
        if threshold_mode == "half_weight":
            self.thresholds = threshold_scale * weights  # shape (n_subjects,)
        elif threshold_mode == "fixed":
            self.thresholds = torch.full_like(weights, threshold_scale)
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
        self.threshold_mode = threshold_mode

        c = env.config
        self._obs_low = np.array([0] * c.n_subjects + [1, 1], dtype=np.float32)
        self._obs_high = np.array(
            [c.max_knowledge] * c.n_subjects + [c.max_energy, c.n_days],
            dtype=np.float32,
        )
        self._obs_range = np.maximum(self._obs_high - self._obs_low, 1e-8)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32) - self._obs_low) / self._obs_range

    def _subject_cvar(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Compute per-subject CVaR from quantiles.

        Args:
            quantiles: shape (batch, n_actions, n_subjects, n_quantiles)

        Returns:
            CVaR per (batch, action, subject), shape (batch, n_actions, n_subjects)
        """
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        return sorted_q[..., : self.cvar_n].mean(dim=-1)

    def _constrained_action(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Lagrangian-relaxed action selection.

        score(s, a) = sum_i E[Z_i(s,a)]
                      - lambda * sum_i [ max(0, T_i - CVaR_alpha(Z_i(s,a))) ]^2

        Quadratic-sum violation penalty: differentiable, smooth, and
        amplifies relative differences in CVaR predictions when all
        actions violate. Combined with Monte-Carlo per-subject return
        targets (rather than Bellman bootstrapping), this avoids the
        chicken-and-egg pathology of low-CVaR-everywhere collapse.

        Args:
            quantiles: shape (batch, n_actions, n_subjects, n_quantiles)
        """
        means = quantiles.mean(dim=-1)
        cvars = self._subject_cvar(quantiles)
        thresh = self.thresholds.view(1, 1, -1)
        violation = torch.clamp(thresh - cvars, min=0.0)
        score = means.sum(dim=-1) - self.lagrangian_lambda * (violation ** 2).sum(dim=-1)
        return score.argmax(dim=-1)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.tensor(
                self._normalize(obs), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            quantiles = self.online_net(state_t)   # (1, A, S, Q)
            return int(self._constrained_action(quantiles).item())

    def _quantile_huber_loss(
        self,
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """Per-subject quantile Huber loss, averaged across subjects.

        Args:
            quantiles: (batch, n_subjects, n_quantiles)
            target_quantiles: (batch, n_subjects, n_quantiles)
        """
        pred = quantiles.unsqueeze(3)              # (batch, S, Q, 1)
        target = target_quantiles.unsqueeze(2)     # (batch, S, 1, Q)
        td_error = target - pred                   # (batch, S, Q, Q)

        huber = torch.where(
            td_error.abs() <= kappa,
            0.5 * td_error.pow(2),
            kappa * (td_error.abs() - 0.5 * kappa),
        )
        tau_hat = self.tau_hat.view(1, 1, -1, 1)   # (1, 1, Q, 1)
        weight = torch.abs(tau_hat - (td_error < 0).float())

        loss = (weight * huber).sum(dim=3).mean(dim=2)  # (batch, S)
        return loss.mean()

    def _update(self) -> float:
        """One gradient step: distributional Bellman update per subject.

        For each subject i, regress current per-subject quantiles toward
            r_i + gamma * theta_i(s', a*(s')) * (1 - done)
        where a*(s') is selected on the online network using the same
        Lagrangian-relaxed score (Double-DQN-style decoupling: select
        with online, evaluate with target).
        """
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
        subj_rewards = torch.tensor(
            np.array([t.subject_rewards for t in batch]),
            dtype=torch.float32, device=self.device,
        )  # (B, S)
        next_states = torch.tensor(
            np.array([self._normalize(t.next_state) for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device,
        )

        all_quantiles = self.online_net(states)
        action_idx = (
            actions.view(-1, 1, 1, 1)
            .expand(-1, 1, self.n_subjects, self.n_quantiles)
        )
        current_quantiles = all_quantiles.gather(1, action_idx).squeeze(1)

        with torch.no_grad():
            next_quantiles_target = self.target_net(next_states)
            next_quantiles_online = self.online_net(next_states)
            best_actions = self._constrained_action(next_quantiles_online)
            best_idx = (
                best_actions.view(-1, 1, 1, 1)
                .expand(-1, 1, self.n_subjects, self.n_quantiles)
            )
            next_best = next_quantiles_target.gather(1, best_idx).squeeze(1)
            target_quantiles = (
                subj_rewards.unsqueeze(2)
                + self.gamma * next_best * (1 - dones.view(-1, 1, 1))
            )

        loss = self._quantile_huber_loss(current_quantiles, target_quantiles)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def _subject_reward_vector(self, exam_scores: dict[int, float]) -> np.ndarray:
        r = np.zeros(self.n_subjects, dtype=np.float32)
        for subj, score in exam_scores.items():
            r[subj] = score
        return r

    def train(
        self,
        n_episodes: int = 5000,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[float]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        episode_returns: list[float] = []
        total_steps = 0
        self.epsilon = self.epsilon_start

        iterator = trange(n_episodes, desc="SC-CVaR-DQN") if verbose else range(n_episodes)
        for ep in iterator:
            ep_seed = int(rng.integers(0, 2**31))
            obs, _ = self.env.reset(seed=ep_seed)
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                subj_r = self._subject_reward_vector(info.get("exam_scores", {}))
                self.buffer.push(
                    SubjectTransition(obs, action, subj_r, next_obs, done)
                )
                self._update()
                total_steps += 1

                if total_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                total_reward += reward
                obs = next_obs

            episode_returns.append(total_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if verbose and ep % 500 == 0:
                recent = (
                    np.mean(episode_returns[-100:])
                    if len(episode_returns) >= 100
                    else np.mean(episode_returns)
                )
                logger.info(
                    "SC-CVaR-DQN ep %d, avg return: %.3f, epsilon: %.3f",
                    ep, recent, self.epsilon,
                )

        return episode_returns

    def evaluate(self, n_episodes: int = 1000, seed: int = 42) -> dict:
        """Evaluate with greedy (no-epsilon) constrained policy.

        Returns metrics including per-subject failure rate (fraction of
        episodes where subject i's exam score fell below its threshold
        T_i), which is the constraint-relevant metric for this agent.
        """
        all_returns: list[float] = []
        all_worst_exam: list[float] = []
        thresholds_np = self.thresholds.detach().cpu().numpy()
        per_subject_failures = np.zeros(self.n_subjects, dtype=np.int64)
        any_failure_count = 0
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

            ep_failed = False
            for i in range(self.n_subjects):
                if exam_scores.get(i, 0.0) < thresholds_np[i]:
                    per_subject_failures[i] += 1
                    ep_failed = True
            if ep_failed:
                any_failure_count += 1

        returns_arr = np.array(all_returns)
        worst_arr = np.array(all_worst_exam)
        cutoff = int(np.ceil(len(returns_arr) * 0.1))
        sorted_returns = np.sort(returns_arr)
        cvar_10 = float(np.mean(sorted_returns[:cutoff]))

        return {
            "mean_total_reward": float(np.mean(returns_arr)),
            "std_total_reward": float(np.std(returns_arr)),
            "mean_worst_exam": float(np.mean(worst_arr)),
            "failure_rate": float(any_failure_count / n_episodes),
            "per_subject_failure_rate": (per_subject_failures / n_episodes).tolist(),
            "cvar_10": cvar_10,
            "all_returns": all_returns,
        }

    def save(self, path: str) -> None:
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.online_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.online_net.state_dict())
