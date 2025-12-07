#!/bin/bash

# Run all additional experiments
# Usage: ./run_experiments.sh [--quick|--full]

cd "$(dirname "$0")/.."

mkdir -p logs/experiments
mkdir -p results/experiments

MODE="${1:---quick}"

WANDB_PROJECT="value-estimation-experiments"
WANDB_ENTITY="syedsaadhasanemad-iba-institute-of-business-administration"
TIMESTEPS=1000000
SEEDS=(0)

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

ENVS=("Hopper-v4" "Walker2d-v4")

for ENV in "${ENVS[@]}"; do
    echo "=========================================="
    echo "Running experiments for Environment: $ENV"
    echo "=========================================="

    for seed in "${SEEDS[@]}"; do
        echo "--- Seed $seed ---"
        
        # Experiment 1: Adaptive Value Steps
        run_experiment "${ENV}_vpg_adaptive" "vpg_adaptive.py" "" $seed
        
        # Experiment 2a: Large Critic (128x128)
        run_experiment "${ENV}_vpg_critic_128" "vpg_large_critic.py" "--critic-sizes 128 128" $seed
        
        # Experiment 2b: XL Critic (256x256)  
        run_experiment "${ENV}_vpg_critic_256" "vpg_large_critic.py" "--critic-sizes 256 256" $seed
        
        # Experiment 3: VPG+ (with clipping)
        run_experiment "${ENV}_vpg_plus" "vpg_plus.py" "--clip-eps 0.2" $seed
        
        # Experiment 4: Alternative Advantage Estimators
        run_experiment "${ENV}_vpg_mc" "vpg_advantage.py" "--advantage-type mc" $seed
        run_experiment "${ENV}_vpg_nstep5" "vpg_advantage.py" "--advantage-type nstep --nstep 5" $seed
        run_experiment "${ENV}_vpg_gae" "vpg_advantage.py" "--advantage-type gae --gae-lambda 0.95" $seed
        
        # Value Step Comparison
        run_experiment "${ENV}_vpg_valstep_50" "../VPG_single_file.py" "--num-value-step 50" $seed
        run_experiment "${ENV}_vpg_valstep_100" "../VPG_single_file.py" "--num-value-step 100" $seed
        
    done
done

echo ""
echo "=== All Experiments Complete ==="
echo "Logs saved to: logs/experiments/"
echo ""
echo "Run 'python experiments/plot_experiments.py' to visualize results"
