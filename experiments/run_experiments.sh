#!/bin/bash

# Run all additional experiments
# Usage: ./run_experiments.sh [--quick|--full]

cd "$(dirname "$0")/.."

mkdir -p logs/experiments
mkdir -p results/experiments

MODE="${1:---quick}"

WANDB_PROJECT="value-estimation-experiments"
WANDB_ENTITY="syedsaadhasanemad-iba-institute-of-business-administration"

if [ "$MODE" = "--quick" ]; then
    TIMESTEPS=500000
    SEEDS=(0)
    echo "=== Quick Mode: 500k steps, 1 seed ==="
else
    TIMESTEPS=1000000
    SEEDS=(0 1 2)
    echo "=== Full Mode: 1M steps, 3 seeds ==="
fi

ENV="Hopper-v4"

run_experiment() {
    local name=$1
    local script=$2
    local extra_args=$3
    local seed=$4
    
    echo "[$(date)] Starting $name (seed=$seed)"
    LOG="logs/experiments/${name}_s${seed}.log"
    
    python "experiments/$script" \
        --env-id $ENV \
        --seed $seed \
        --total-timesteps $TIMESTEPS \
        --wandb-project-name $WANDB_PROJECT \
        --wandb-entity "$WANDB_ENTITY" \
        $extra_args \
        2>&1 | tee "$LOG"
    
    echo "[$(date)] Completed $name (seed=$seed)"
}

export -f run_experiment
export ENV TIMESTEPS WANDB_PROJECT WANDB_ENTITY

echo ""
echo "=== Running Additional Experiments ==="
echo "Environment: $ENV"
echo "Timesteps: $TIMESTEPS"
echo ""

for seed in "${SEEDS[@]}"; do
    echo "--- Seed $seed ---"
    
    # Experiment 1: Adaptive Value Steps
    run_experiment "vpg_adaptive" "vpg_adaptive.py" "" $seed
    
    # Experiment 2a: Large Critic (128x128)
    run_experiment "vpg_critic_128" "vpg_large_critic.py" "--critic-sizes 128 128" $seed
    
    # Experiment 2b: XL Critic (256x256)  
    run_experiment "vpg_critic_256" "vpg_large_critic.py" "--critic-sizes 256 256" $seed
    
    # Experiment 3: VPG+ (with clipping)
    run_experiment "vpg_plus" "vpg_plus.py" "--clip-eps 0.2" $seed
    
    # Experiment 4: Alternative Advantage Estimators
    run_experiment "vpg_mc" "vpg_advantage.py" "--advantage-type mc" $seed
    run_experiment "vpg_nstep5" "vpg_advantage.py" "--advantage-type nstep --nstep 5" $seed
    run_experiment "vpg_gae" "vpg_advantage.py" "--advantage-type gae --gae-lambda 0.95" $seed
    
    # Baseline: VPG with 50 value steps (for comparison)
    run_experiment "vpg_baseline" "../VPG_single_file.py" "--num-value-step 50" $seed
    
done

echo ""
echo "=== All Experiments Complete ==="
echo "Logs saved to: logs/experiments/"
echo ""
echo "Run 'python experiments/plot_experiments.py' to visualize results"
