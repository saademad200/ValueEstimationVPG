#!/bin/bash

# Quick test for PPO to verify it works before full replication

mkdir -p logs
mkdir -p results

# Add project root to PYTHONPATH for local tianshou
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

TOTAL_TIMESTEPS=100000  # 100k for quick test
NUM_ENVS=16
PPO_NUM_STEPS=2048

WANDB_PROJECT="value-estimation-vpg-test"
WANDB_ENTITY="syedsaadhasanemad-iba-institute-of-business-administration"

EPOCHS=$((TOTAL_TIMESTEPS / PPO_NUM_STEPS))

echo "=== Quick Test: PPO on Hopper-v4 ==="
echo "Epochs: $EPOCHS (from $TOTAL_TIMESTEPS / $PPO_NUM_STEPS)"
echo "PYTHONPATH includes local tianshou: $(pwd)"
echo "This should take ~1-2 minutes..."

cd examples/mujoco

python mujoco_ppo.py \
    --task Hopper-v4 \
    --seed 0 \
    --gae-lambda 0.95 \
    --gamma 0.99 \
    --step-per-collect $PPO_NUM_STEPS \
    --training-num $NUM_ENVS \
    --epoch $EPOCHS \
    --wandb-project $WANDB_PROJECT \
    --wandb-entity $WANDB_ENTITY \
    2>&1 | tee ../../logs/quick_test_ppo.log

cd ../..

echo ""
echo "=== PPO Quick Test Complete ==="
echo "Check logs/quick_test_ppo.log for 'Final reward' output"
