#!/bin/bash

# Run all additional experiments
# Usage: ./run_experiments.sh [--quick|--full]

cd "$(dirname "$0")/.."

mkdir -p logs/experiments
mkdir -p results/experiments

MODE="${1:---quick}"

WANDB_PROJECT="value-estimation-experiments-v2"
WANDB_ENTITY="syedsaadhasanemad-iba-institute-of-business-administration"
TIMESTEPS=1000000
SEEDS=(0)
MAX_PARALLEL=4

run_experiment() {
    local name=$1
    local script=$2
    local extra_args=$3
    local seed=$4
    local env_id=$5
    
    echo "[$(date)] Starting $name (seed=$seed)"
    LOG="logs/experiments/${name}_s${seed}.log"
    
    python "experiments/$script" \
        --env-id $env_id \
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
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_adaptive" "vpg_adaptive.py" "" $seed "$ENV" &
        
        # Experiment 2a: Large Critic (128x128)
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_critic_128" "vpg_large_critic.py" "--critic-sizes 128 128" $seed "$ENV" &
        
        # Experiment 2b: XL Critic (256x256)  
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_critic_256" "vpg_large_critic.py" "--critic-sizes 256 256" $seed "$ENV" &
        
        # Experiment 3: VPG+ (with clipping)
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_plus" "vpg_plus.py" "--clip-eps 0.2" $seed "$ENV" &
        
        # Experiment 4: Alternative Advantage Estimators
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_mc" "vpg_advantage.py" "--advantage-type mc" $seed "$ENV" &
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_nstep5" "vpg_advantage.py" "--advantage-type nstep --nstep 5" $seed "$ENV" &
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_gae" "vpg_advantage.py" "--advantage-type gae --gae-lambda 0.95" $seed "$ENV" &
        
        # Experiment 4b: Normalized GAE (New "Better" Advantage)
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_gaenorm" "vpg_advantage.py" "--advantage-type gae --gae-lambda 0.95 --norm-adv" $seed "$ENV" &

        # Value Step Comparison
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_valstep_50" "../VPG_single_file.py" "--num-value-step 50" $seed "$ENV" &
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_vpg_valstep_100" "../VPG_single_file.py" "--num-value-step 100" $seed "$ENV" &

        # --- New Combinations (vpg_combo.py) ---

        # 1. Clipped + Normalized (Equivalent to VPG+)
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_combo_clip_norm" "vpg_combo.py" "--clip-eps 0.2 --norm-adv" $seed "$ENV" &

        # 2. Clipping + Adaptive
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_combo_clip_adaptive" "vpg_combo.py" "--clip-eps 0.2 --adaptive-value-steps" $seed "$ENV" &

        # 3. Clipping + Adaptive + Normalized
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_combo_clip_adaptive_norm" "vpg_combo.py" "--clip-eps 0.2 --adaptive-value-steps --norm-adv" $seed "$ENV" &

        # 4. MC + Adaptive
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        run_experiment "${ENV}_combo_mc_adaptive" "vpg_combo.py" "--advantage-type mc --adaptive-value-steps" $seed "$ENV" &

        
    done
done

wait


echo ""
echo "=== All Experiments Complete ==="
echo "Logs saved to: logs/experiments/"
echo ""
echo "Run 'python experiments/plot_experiments.py' to visualize results"
