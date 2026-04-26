# CVaR-DQN Exam Scheduler

Risk-sensitive reinforcement learning for optimal exam study scheduling. Uses CVaR-DQN (distributional RL) to find study strategies that avoid failing any single exam.

Course project for "Reinforcement Learning and Decision Making Under Uncertainty" — University of Neuchatel, Spring 2026.

## Setup

```bash
python3 -m venv venv
source venvironment/bin/activate
pip install -r requirements.txt
```

## Run Experiments

### 1. Main comparison (all agents, all configs)

```bash
# Full run (~40 min) — 7 agents, 3 configs, 5 seeds
python experiments/run_all.py --configs small medium large

# Quick test (~4 min) — single seed, fewer episodes
python experiments/run_all.py --configs small --n-train 1000 --seeds 42

# Medium only
python experiments/run_all.py --configs medium --n-train 5000 --n-eval 1000
```

### 2. Ablation studies (~2 hours)

```bash
# 9 ablation variants on medium config, DQN vs CVaR-DQN
python experiments/ablation.py

# Faster with fewer episodes
python experiments/ablation.py --n-train 2000 --n-eval 500
```

### 3. Robustness tests (~20 min)

```bash
# Train on normal conditions, test under perturbations
python experiments/robustness.py

# Faster
python experiments/robustness.py --n-train 2000 --n-eval 500
```

### 4. Generate plots

```bash
# Generates all figures from CSV results
python experiments/plot_results.py
```

### Run everything at once

```bash
python experiments/run_all.py --configs small medium large
python experiments/ablation.py
python experiments/robustness.py
python experiments/plot_results.py
```

## CLI Options

All experiment scripts support these arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-train` | 5000 | Training episodes per agent |
| `--n-eval` | 1000 | Evaluation episodes |
| `--results-dir` | `results` | Output directory for CSVs |
| `--seeds` | 42 123 456 789 1024 | Random seeds |
| `--configs` | small medium large | Environment sizes (run_all.py only) |

## Output

- `results/*.csv` — raw experiment data
- `results/plots/*.png` — comparison charts, learning curves, ablation, robustness

## Project Structure

```
environment/study_env.py           # Gymnasium environment
agents/heuristics.py       # Uniform, Most-Urgent-First, Lowest-Knowledge-First
agents/value_iteration.py  # Exact DP (small config only)
agents/q_learning.py       # Tabular Q-learning (small/medium)
agents/dqn.py              # DQN + Double DQN (all configs)
agents/cvar_dqn.py         # CVaR-DQN — main contribution
experiments/run_all.py     # Main comparison experiments
experiments/ablation.py    # Ablation studies
experiments/robustness.py  # Robustness tests
experiments/plot_results.py # Plot generation
```
