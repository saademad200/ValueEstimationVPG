#!/bin/bash

# Experiment script to replicate "Improving Value Estimation Critically Enhances VPG" paper
# Table 2: VPG vs PPO with different GAE lambda values

mkdir -p logs
mkdir -p results

# Add project root to PYTHONPATH for local tianshou
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Paper settings
TOTAL_TIMESTEPS_HOPPER_WALKER=5000000  # 5M for Hopper and Walker
TOTAL_TIMESTEPS_HALFCHEETAH=10000000   # 10M for HalfCheetah
NUM_ENVS=64
VPG_NUM_STEPS=32
PPO_NUM_STEPS=2048
MAX_PARALLEL=2

WANDB_PROJECT="value-estimation-replication"
WANDB_ENTITY="syedsaadhasanemad-iba-institute-of-business-administration"

# Paper uses 5 seeds for statistical significance
SEEDS=(0 1 2 3 4)

# Environments from Table 2
ENVS=("Hopper-v4" "Walker2d-v4" "HalfCheetah-v4")

# GAE lambda values from Table 2
GAE_LAMBDAS=("0.95" "1.0")

# Value steps to test (paper uses 50 as default for enhanced VPG)
VALUE_STEPS=(1 10 50 100)

RESULTS_FILE="results/summary_results.csv"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "algorithm,gae_lambda,value_steps,env,seed,mean_return,std_return" > "$RESULTS_FILE"
fi

get_timesteps() {
    local env=$1
    if [ "$env" = "HalfCheetah-v4" ]; then
        echo $TOTAL_TIMESTEPS_HALFCHEETAH
    else
        echo $TOTAL_TIMESTEPS_HOPPER_WALKER
    fi
}

run_vpg_experiment() {
    local gae_lambda=$1
    local value_steps=$2
    local env=$3
    local seed=$4

    local timesteps=$(get_timesteps "$env")
    
    EXTRA_ARGS="--env-id $env --seed $seed --num-value-step $value_steps --gae-lambda $gae_lambda --gamma 0.99 --num-steps $VPG_NUM_STEPS --num-envs $NUM_ENVS --total-timesteps $timesteps --wandb-project-name $WANDB_PROJECT --wandb-entity $WANDB_ENTITY"

    echo "[$(date)] Starting VPG: env=$env, gae_lambda=$gae_lambda, value_steps=$value_steps, seed=$seed"

    LOG_FILE="logs/VPG_${env}_gae${gae_lambda}_vs${value_steps}_s${seed}.log"
    python VPG_single_file.py $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"

    # Extract final cumulative reward
    local mean_return=$(grep -oP 'Cumulative Reward: \K[0-9.]+' "$LOG_FILE" | tail -1)
    if [ -n "$mean_return" ]; then
        echo "VPG,$gae_lambda,$value_steps,$env,$seed,$mean_return,0.0" >> "$RESULTS_FILE"
    else
        echo "VPG,$gae_lambda,$value_steps,$env,$seed,NaN,0.0" >> "$RESULTS_FILE"
    fi
}

run_ppo_experiment() {
    local gae_lambda=$1
    local env=$2
    local seed=$3

    local timesteps=$(get_timesteps "$env")
    local epochs=$((timesteps / 10240))  # Match VPG's ~10k steps per test
    
    EXTRA_ARGS="--task $env --seed $seed --gae-lambda $gae_lambda --gamma 0.99 --step-per-collect 2048 --step-per-epoch 10240 --training-num $NUM_ENVS --epoch $epochs --batch-size 64 --wandb-project $WANDB_PROJECT --wandb-entity $WANDB_ENTITY"

    echo "[$(date)] Starting PPO: env=$env, gae_lambda=$gae_lambda, seed=$seed"

    LOG_FILE="logs/PPO_${env}_gae${gae_lambda}_s${seed}.log"
    python examples/mujoco/mujoco_ppo.py $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"

    # Extract final reward
    local mean_return=$(grep -oP "Final reward: \K[0-9.]+" "$LOG_FILE" | tail -1)
    if [ -n "$mean_return" ]; then
        echo "PPO,$gae_lambda,10,$env,$seed,$mean_return,0.0" >> "$RESULTS_FILE"
    else
        echo "PPO,$gae_lambda,10,$env,$seed,NaN,0.0" >> "$RESULTS_FILE"
    fi
}

export -f run_vpg_experiment run_ppo_experiment get_timesteps
export WANDB_PROJECT WANDB_ENTITY VPG_NUM_STEPS PPO_NUM_STEPS NUM_ENVS RESULTS_FILE
export TOTAL_TIMESTEPS_HOPPER_WALKER TOTAL_TIMESTEPS_HALFCHEETAH

cleanup() {
    echo "Killing all background processes..."
    pkill -P $$
    exit 1
}

trap cleanup SIGINT SIGTERM

echo "=== Starting Paper Replication Experiments ==="
echo "Target: Table 2 - VPG vs PPO with GAE lambda 0.95 and 1.0"
echo ""

for env in "${ENVS[@]}"; do
    for gae_lambda in "${GAE_LAMBDAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Run VPG with value_steps=50 (paper's recommended setting)
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
            run_vpg_experiment "$gae_lambda" 50 "$env" "$seed" &

            # Run PPO baseline
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
            run_ppo_experiment "$gae_lambda" "$env" "$seed" &
        done
    done
done

wait

echo -e "\n=== Final Results ==="
echo "Results saved to $RESULTS_FILE"

# Generate summary statistics
python3 << 'EOF'
import pandas as pd
import numpy as np

df = pd.read_csv("results/summary_results.csv")
summary = df.groupby(['algorithm', 'gae_lambda', 'env']).agg({
    'mean_return': ['mean', 'std']
}).round(2)
print("\n=== Summary Statistics ===")
print(summary.to_string())
EOF

echo -e "\nAll experiments completed!"
