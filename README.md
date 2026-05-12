# CVaR-DQN Exam Scheduler

Risk-sensitive reinforcement learning for optimal exam study scheduling. Compares DQN, CVaR-DQN (distributional RL with worst-case-aware action selection), and a pilot extension SC-CVaR-DQN (multi-objective distributional constrained MDP) against value-iteration and heuristic baselines.

Course project for *Reinforcement Learning and Decision Making Under Uncertainty* — University of Neuchâtel, Spring 2026.

Project report: [`report/report_formal.pdf`](report/report_formal.pdf).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

### 1. Main comparison (all agents, all configs)

```bash
# Full run (~90 min on CPU) — 8 agents, 3 configs, 5 seeds
python -m experiments.run_all --configs small medium large

# Quick test (~4 min) — single seed, fewer episodes
python -m experiments.run_all --configs small --n-train 1000 --seeds 42

# Medium config only
python -m experiments.run_all --configs medium --n-train 5000 --n-eval 1000
```

### 2. Ablation studies (~3 hours sequential)

```bash
# 9 ablation variants on medium config, DQN vs CVaR-DQN vs SC-CVaR-DQN
python -m experiments.ablation

# Faster with fewer episodes
python -m experiments.ablation --n-train 2000 --n-eval 500
```

### 3. Robustness tests (~30 min)

```bash
# Train on normal conditions, test under perturbations
python -m experiments.robustness

# Faster
python -m experiments.robustness --n-train 2000 --n-eval 500
```

### 4. Generate plots

```bash
# Generates all figures from CSV results in results/
python -m experiments.plot_results
```

### Run everything

```bash
python -m experiments.run_all --configs small medium large
python -m experiments.ablation
python -m experiments.robustness
python -m experiments.plot_results
```

## CLI Options

All experiment scripts support these arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-train` | 5000 | Training episodes per agent |
| `--n-eval` | 1000 | Evaluation episodes |
| `--results-dir` | `results` | Output directory for CSVs |
| `--seeds` | 42 123 456 789 1024 | Random seeds |
| `--configs` | small medium large | Environment sizes (`run_all` only) |

## Output

- `results/*.csv` — raw experiment data (one row per agent × seed × config)
- `results/plots/*.png` — comparison charts, learning curves, per-subject failure rate, ablation, robustness, radar

## Project Structure

```
environment/study_env.py        # Gymnasium environment (MDP definition)
agents/heuristics.py            # Uniform, Most-Urgent-First, Lowest-Knowledge-First
agents/value_iteration.py       # Exact dynamic programming (small config only)
agents/q_learning.py            # Tabular Q-learning (small / medium)
agents/dqn.py                   # DQN + Double DQN (all configs)
agents/cvar_dqn.py              # CVaR-DQN — main contribution
agents/subject_cvar_dqn.py      # SC-CVaR-DQN — pilot extension
experiments/run_all.py          # Main agent comparison
experiments/ablation.py         # Ablation studies on medium config
experiments/robustness.py       # Distribution-shift tests
experiments/plot_results.py     # Generate all plots from CSVs
report/report_formal.tex        # Project report (LaTeX source)
```

## License

MIT — see [LICENSE](LICENSE).
