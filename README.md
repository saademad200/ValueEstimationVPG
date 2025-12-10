# Reproduction and Extension of "Improving Value Estimation Critically Enhances Vanilla Policy Gradient"

This repository contains our reproduction and novel extensions of the paper **"Improving Value Estimation Critically Enhances Vanilla Policy Gradient"** (Wang et al., ICML 2025) [[arXiv]](https://arxiv.org/abs/2505.19247) [[Original Code]](https://github.com/saademad200/ValueEstimationVPG.git).

**Course**: CSE 682 - Reinforcement Learning  
**Instructor**: Syed Ali Raza  
**Institution**: Institute of Business Administration, Karachi  
**Term**: Fall 2025

## Overview

This project reproduces and extends the key findings from Wang et al. (2025), which demonstrates that **improved value estimation** (via more value update steps per iteration) is the critical factor enabling Vanilla Policy Gradient (VPG) to match or exceed PPO performance—not trust region constraints.

## Repository Structure

```
├── VPG_single_file.py          # Base VPG implementation (adapted from CleanRL)
├── PPO_single_file.py          # PPO baseline implementation
├── run_experiment.sh           # Paper replication script (VPG vs PPO)
├── experiments/                # Novel improvement experiments
│   ├── run_experiments.sh      # Script to run all novel experiments
│   ├── vpg_adaptive.py         # Adaptive value step scheduling
│   ├── vpg_large_critic.py     # Enlarged critic architectures
│   ├── vpg_plus.py             # VPG with PPO clipping
│   ├── vpg_advantage.py        # Alternative advantage estimators
│   └── vpg_combo.py            # Combination experiments
├── results/                    # All experimental results (CSV files)
├── paper.pdf                   # Original paper
```

## Computational Setup

- **GPU**: NVIDIA RTX 4070
- **Tracking**: [Weights & Biases](https://wandb.ai) for experiment logging
- **Environments**: MuJoCo via [Gymnasium](https://gymnasium.farama.org/)
  - Hopper-v4 (11-dim state, 3-dim action)
  - Walker2d-v4 (17-dim state, 6-dim action)
- **Training**: 20% of paper's total timesteps due to compute constraints
  - Hopper/Walker2d: 200K timesteps (paper uses 1M)

## Installation

```bash
# Clone repository
git clone https://github.com/saademad200/ValueEstimationVPG.git
cd ValueEstimationVPG

# Install dependencies
pip install -r requirements.txt

# Required packages: gymnasium[mujoco], torch, numpy, wandb, tyro
```

## Running Experiments

### Paper Replication (VPG vs PPO)

```bash
# Run replication experiments
./run_experiment.sh

# Example single run
python VPG_single_file.py --env_id Hopper-v4 --seed 0 --num_value_step 50
```

### Novel Improvement Experiments

```bash
# Run all novel experiments
./experiments/run_experiments.sh

# Individual experiments
python experiments/vpg_adaptive.py --env-id Hopper-v4 --seed 0
python experiments/vpg_large_critic.py --env-id Hopper-v4 --critic-sizes 128 128
python experiments/vpg_plus.py --env-id Walker2d-v4 --clip-eps 0.2
python experiments/vpg_advantage.py --advantage-type gae --norm-adv
python experiments/vpg_combo.py --advantage-type mc --adaptive-value-steps
```

## Experiment Overview

### Baseline Configurations
- **Paper Replication**: VPG with 50 value steps (baseline) vs PPO
- **Novel Experiments**: VPG with 100 value steps as baseline for adaptive comparisons

### Novel Experiments

| Experiment | Script | Key Findings |
|------------|--------|--------------|
| **Adaptive Value Steps** | `vpg_adaptive.py` | ✅ Same results as 100 value steps, **less processing time** (avg 97 value steps declining over training) |
| **Large Critic** | `vpg_large_critic.py` | ❌ No noticeable improvement with [128,128] or [256,256] vs [64,64] |
| **VPG+** | `vpg_plus.py` | ✅ Slightly improved results with PPO clipping |
| **Monte Carlo** | `vpg_advantage.py --advantage-type mc` | ⚪ Similar performance to GAE |
| **Normalized GAE** | `vpg_advantage.py --norm-adv` | ✅ Slightly better than baseline |
| **n-step Returns** | `vpg_advantage.py --advantage-type nstep` | ❌ Very poor results |
| **MC + Adaptive** | `vpg_combo.py` | ⭐ **Best overall results** |

### Combination Results

Surprisingly, combining individually successful improvements did **not** yield synergistic benefits:

| Combination | Result |
|-------------|--------|
| Clipped + Normalized | ❌ Worse than individuals |
| Clipping + Adaptive | ❌ Worse than individuals |
| Clipping + Adaptive + Normalized | ❌ Worse than individuals |
| **MC + Adaptive** | ⭐ Best results out of all |

### Key Finding: Adaptive Value Steps Success

- VPG Adaptive and MC + Adaptive completed in **similar time** as VPG with 100 value steps
- Cumulative reward graphs were nearly identical across all three methods
- **Adaptive scheduling is a major success**: achieves same performance with less computation
- With full training (1M timesteps), time savings would be even more pronounced as average value steps continued declining

## Results

All experiment results are stored in `results/`:
- Learning curve CSVs for each experiment configuration
- Summary statistics in `results/summary_results.csv`

## References

```bibtex
@inproceedings{
  wang2025improving,
  title={Improving Value Estimation Critically Enhances Vanilla Policy Gradient},
  author={Tao Wang and Ruipeng Zhang and Sicun Gao},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

### Implementation Sources
- **CleanRL**: [https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) - Base VPG/PPO implementations
- **Tianshou**: [https://github.com/thu-ml/tianshou](https://github.com/thu-ml/tianshou) - MuJoCo experiment framework
- **Gymnasium**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/) - RL environments
- **Weights & Biases**: [https://wandb.ai](https://wandb.ai) - Experiment tracking

## License

This project is for academic purposes as part of CSE 682 coursework at IBA Karachi.
