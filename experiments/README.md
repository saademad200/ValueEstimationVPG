# Experiments for Value Estimation VPG

This directory contains experiments extending the "Improving Value Estimation Critically Enhances Vanilla Policy Gradient" paper.

## Design Philosophy
These experiments are designed to be **minimal modifications** of the base `VPG_single_file.py` implementation.
- **Base:** The original `VPG_single_file.py` remains the "gold standard" reference.
- **Modifications:** Each experiment script (`vpg_*.py`) copies the base logic and modifies **only** the specific components relevant to the hypothesis.
- **Goal:** To isolate the effect of specific changes (e.g., value steps, network size, advantage methods) without confounding variables.

---

## Experiments Overview

| Experiment | Script | Goal | Implementation Change from Base |
|------------|--------|------|---------------------------------|
| **Adaptive Value Steps** | `vpg_adaptive.py` | Auto-tune value limit | Adds `get_adaptive_value_steps()` to change `num_value_steps` dynamically based on `value_loss`. |
| **Critic Architecture** | `vpg_large_critic.py` | Test network capacity | Modification to `Agent` class to accept configurable `critic_sizes` (e.g. `[128,128]`) instead of fixed `[64,64]`. |
| **VPG+** | `vpg_plus.py` | Combine VPG with Clipping | Adds PPO-style `ratio` calculation and `clip_grad_norm_` logic to the policy loss. |
| **Advantage Estimators** | `vpg_advantage.py` | Compare estimators | Replaces default GAE with optional `compute_mc_returns` or `compute_nstep_returns`. |

### 1. Adaptive Value Steps (`vpg_adaptive.py`)
- **Hypothesis:** We can save compute by using fewer value steps when the value function is already accurate.
- **Implementation:**
    - **Base:** Constant `num_value_step=50`.
    - **New:** Checks `value_loss` after updates. If high (>0.5), increases steps (up to 100). If low (<0.1), decreases steps (down to 10).
- **Expectation:** Similar final reward to baseline, but with lower "Cumulative Value Steps" (tracked in charts).

### 2. Critic Architecture (`vpg_large_critic.py`)
- **Hypothesis:** Value estimation is the bottleneck. A larger critic network should learn the value function faster/better.
- **Implementation:**
    - **Base:** Fixed 64x64 MLP for both actor and critic.
    - **New:** `Agent` init accepts `critic_sizes` list. We test `[128,128]` and `[256,256]`.
- **Expectation:** potentially faster convergence or higher final reward on complex envs.

### 3. VPG+ (VPG with PPO Clipping) (`vpg_plus.py`)
- **Hypothesis:** High value steps improve VPG, but large policy updates can still destabilize it. PPO clipping prevents this.
- **Implementation:**
    - **Base:** Standard policy gradient `-(log_prob * advantage).mean()`.
    - **New:** Calculates `ratio = (new_log_prob - old_log_prob).exp()` and applies PPO's `min(surr1, surr2)` objective.
- **Expectation:** More stable learning curves (less variance/crashes) than standard VPG.

### 4. Alternative Advantage Estimators (`experiments/vpg_advantage.py`)
This script compares different methods for estimating the advantage function $A(s,a)$.

**Methods:**
- **Monte Carlo (MC):** Uses the full discounted return $G_t$ as the target. Zero bias, high variance.
- **n-step Return:** Bootstraps after $n$ steps. configurable bias-variance trade-off.
- **GAE (Generalized Advantage Estimation):** Exponentially weighted average of n-step returns. Often the best balance.
- **Normalized GAE (`vpg_gaenorm`):** Applies standard normalization (whitening) to the calculated GAE advantages. $(A - \mu) / \sigma$.

**Hypothesis:** 
- GAE should generally outperform MC in sample efficiency.
- **Normalized GAE** often provides significant stability improvements and faster convergence compared to vanilla GAE, especially in environments with varying reward scales.

**Implementation:**
- Advantage estimation logic is decoupled from the main update loop.
- `vpg_gaenorm` adds a normalization step: `adv = (adv - mean) / (std + 1e-8)`.

**Expectation:** `vpg_gaenorm` > `vpg_gae` > `vpg_nstep` > `vpg_mc`.

### 5. VPG-Hybrid (`experiments/vpg_hybrid.py`)
This experiment represents the culmination of this study, proposing a hybrid algorithm that integrates all improved components. A key paper contribution would be "H-VPG".

**Combined Features:**
1.  **Normalized GAE:** Derived from `vpg_gaenorm`. Stabilizes advantages.
2.  **Large Critic Network:** Uses the [128, 128] architecture from `vpg_large_critic`.
3.  **PPO-style Clipping:** Incorporates the clipping mechanism from `vpg_plus` to prevent catastrophic updates.
4.  **Adaptive Value Steps:** Uses the dynamic step sizing from `vpg_adaptive` to optimize compute efficiency.

**Hypothesis:** This hybrid algorithm is expected to offer the best trade-off between final performance, stability, and training efficiency, outperforming all individual variants.

---

## Running Experiments

To run all experiments (including the new Hybrid VPG), execute:
```bash
./experiments/run_experiments.sh
```

## Plotting Results

To visualize the results, run:
```bash
python experiments/plot_experiments.py
```
This will generate:
1.  **Performance Comparison:** Bar chart of final returns for all variants.
2.  **Compute Efficiency:** Scatter plot of Performance vs. Wall-clock time.
3.  **Time Comparison:** Bar chart of execution time.
4.  **Adaptive Steps:** Analysis of value steps used by the adaptive method.
5.  **Summary Table:** Markdown table with mean/std returns and timing.

Plots are saved in `results/experiments/` organized by environment (e.g., `results/experiments/Hopper-v4/`).
