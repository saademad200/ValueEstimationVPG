# Additional Experiments for Value Estimation VPG

This directory contains novel experiments that extend the findings from **"Improving Value Estimation Critically Enhances Vanilla Policy Gradient"**.

## Experiments Overview

| Experiment | Script | Key Insight |
|------------|--------|-------------|
| Adaptive Value Steps | `vpg_adaptive.py` | Dynamically adjust value steps based on training progress |
| Critic Architecture | `vpg_large_critic.py` | Test if larger critic networks amplify value estimation benefits |
| VPG+ (with clipping) | `vpg_plus.py` | Combine value estimation improvements with PPO-style clipping |
| Advantage Estimators | `vpg_advantage.py` | Compare GAE vs Monte Carlo vs n-step with accurate V |

---

## Experiment 1: Adaptive Value Steps

### Motivation
The paper uses fixed value steps (e.g., 50). But the optimal number may vary during training:
- **Early training**: Value function is inaccurate → more steps needed
- **Late training**: Value function is stable → fewer steps save compute

### Hypothesis
> **Adaptive scheduling can achieve similar performance to fixed high value steps while reducing computational cost by 30-50%.**

### How it Works
```
if value_loss > high_threshold:     # Value function struggling
    value_steps = 100               # More updates
elif value_loss < low_threshold:    # Value function accurate
    value_steps = 10                # Fewer updates
else:
    value_steps = 50                # Default
```

### Run
```bash
python experiments/vpg_adaptive.py --env-id Hopper-v4 --seed 0
```

---

## Experiment 2: Critic Architecture Ablation

### Motivation
If value estimation is the critical factor, does a **larger critic network** further improve performance? The original paper uses 64x64 for both actor and critic.

### Hypothesis
> **A larger critic (128x128) with more capacity for value estimation will outperform the baseline, especially on complex tasks.**

### Configurations
| Variant | Actor Size | Critic Size | 
|---------|------------|-------------|
| Baseline | 64, 64 | 64, 64 |
| Large Critic | 64, 64 | 128, 128 |
| XL Critic | 64, 64 | 256, 256 |

### Run
```bash
python experiments/vpg_large_critic.py --env-id Hopper-v4 --critic-sizes 128 128 --seed 0
```

---

## Experiment 3: VPG+ (VPG with PPO Clipping)

### Motivation
The paper shows value estimation matters more than trust regions (PPO clipping). But what if we combine both? Could we get the best of both worlds?

### Hypothesis
> **VPG with enhanced value estimation PLUS PPO-style clipping (VPG+) will achieve better stability and performance than either method alone.**

### Implementation
- Base VPG with 50 value steps (paper's recommendation)
- Add PPO-style clipping with ε=0.2
- Importance sampling ratio to enable clipping

### Run
```bash
python experiments/vpg_plus.py --env-id Hopper-v4 --clip-eps 0.2 --num-value-step 50 --seed 0
```

---

## Experiment 4: Alternative Advantage Estimators

### Motivation
The paper shows that better value estimation improves VPG. But the advantage function `A(s,a) = Q(s,a) - V(s)` directly uses V. Different advantage estimators trade off bias vs. variance differently.

### Hypothesis
> **With accurate value estimates (50 value steps), GAE should outperform Monte Carlo because bootstrapping with accurate V reduces variance without adding significant bias.**

### Advantage Estimators

| Method | Formula | Bias | Variance |
|--------|---------|------|----------|
| **Monte Carlo** | `R_t - V(s_t)` | Zero | High |
| **n-step (n=5)** | `Σγʳ + γⁿV(s_{t+n}) - V(s_t)` | Medium | Medium |
| **GAE (λ=0.95)** | Exponentially weighted TD errors | Low | Low |

### Run
```bash
# Monte Carlo advantage
python experiments/vpg_advantage.py --env-id Hopper-v4 --advantage-type mc --seed 0

# 5-step return
python experiments/vpg_advantage.py --env-id Hopper-v4 --advantage-type nstep --nstep 5 --seed 0

# GAE (baseline)
python experiments/vpg_advantage.py --env-id Hopper-v4 --advantage-type gae --gae-lambda 0.95 --seed 0
```

---

## Running All Experiments

```bash
# Quick test (100k steps)
./experiments/run_experiments.sh --quick

# Full experiments (5M steps, 3 seeds)
./experiments/run_experiments.sh --full
```

## Visualizing Results

```bash
python experiments/plot_experiments.py --results-dir results/experiments
```

This generates:
- Learning curves comparing all variants
- Bar chart of final performance
- Computational efficiency analysis
