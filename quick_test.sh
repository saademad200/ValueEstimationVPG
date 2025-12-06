#!/bin/bash

# Quick test script - runs a short experiment to verify setup
# Use this before running full paper replication

mkdir -p logs
mkdir -p results

TOTAL_TIMESTEPS=100000  # 100k for quick test
NUM_ENVS=16
VPG_NUM_STEPS=32

WANDB_PROJECT="value-estimation-vpg-test"
WANDB_ENTITY="syedsaadhasanemad-iba-institute-of-business-administration"

RESULTS_FILE="results/quick_test_results.csv"

echo "algorithm,gae_lambda,value_steps,env,seed,mean_return" > "$RESULTS_FILE"

echo "=== Quick Test: VPG with value_steps=50 on Hopper-v4 ==="
echo "This should take ~1-2 minutes..."

python VPG_single_file.py \
    --env-id Hopper-v4 \
    --seed 0 \
    --num-value-step 50 \
    --gae-lambda 0.95 \
    --gamma 0.99 \
    --num-steps $VPG_NUM_STEPS \
    --num-envs $NUM_ENVS \
    --total-timesteps $TOTAL_TIMESTEPS \
    --wandb-project-name $WANDB_PROJECT \
    --wandb-entity $WANDB_ENTITY \
    2>&1 | tee logs/quick_test_vpg.log

# Extract result
MEAN_RETURN=$(grep -oP 'Cumulative Reward: \K[0-9.]+' logs/quick_test_vpg.log | tail -1)
echo "VPG,0.95,50,Hopper-v4,0,$MEAN_RETURN" >> "$RESULTS_FILE"

echo ""
echo "=== Quick Test Complete ==="
echo "Final return: $MEAN_RETURN"
echo "Results saved to $RESULTS_FILE"
echo ""
echo "If this worked, run './run_experiment.sh' for full replication"
