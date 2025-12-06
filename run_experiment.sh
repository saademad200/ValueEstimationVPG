#!/bin/bash

mkdir -p logs
mkdir -p results

TOTAL_TIMESTEPS=100000
NUM_ENVS=64
VPG_NUM_STEPS=32
PPO_NUM_STEPS=2048
MAX_PARALLEL=2

WANDB_PROJECT="mujoco-orig"
WANDB_ENTITY="syedsaadhasanemad-iba-institute-of-business-administration"

ENVS=("Hopper-v4" "Walker2d-v4" "HalfCheetah-v4")
RESULTS_FILE="results/summary_results.csv"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "algorithm,gamma,value_steps,env,mean_return,std_return" > "$RESULTS_FILE"
fi

run_experiment() {
    local algorithm=$1
    local gamma=$2
    local value_steps=$3
    local env=$4

    if [ "$algorithm" = "VPG" ]; then
        SCRIPT="VPG_single_file.py"
        NUM_STEPS=$VPG_NUM_STEPS
        ACTOR_LR="3e-4"
        CRITIC_LR="1e-3"
        EXTRA_ARGS="--actor-learning-rate $ACTOR_LR --critic-learning-rate $CRITIC_LR --num-value-step $value_steps --gamma $gamma --num-steps $NUM_STEPS --num-envs $NUM_ENVS --total-timesteps $TOTAL_TIMESTEPS"
    else
        SCRIPT="examples/mujoco/mujoco_ppo.py"
        NUM_STEPS=$PPO_NUM_STEPS
        # compute epochs from total timesteps and step-per-collect
        EPOCHS=$((TOTAL_TIMESTEPS / NUM_STEPS))
        EXTRA_ARGS="--task $env --step-per-collect $NUM_STEPS --training-num $NUM_ENVS --gamma $gamma --epoch $EPOCHS --wandb-project $WANDB_PROJECT --resume-id $WANDB_ENTITY"
    fi

    echo "[$(date)] Starting $algorithm with gamma=$gamma, value_steps=$value_steps on $env"

    LOG_FILE="logs/${algorithm}_gamma${gamma}_vs${value_steps}_${env}.log"
    python "$SCRIPT" $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"

    # Extract mean return from log (look for 'Average Episode Reward' line)
    local mean_return=$(grep -oP 'Average Episode Reward: \K[0-9.]+' "$LOG_FILE" | tail -1)
    if [ -n "$mean_return" ]; then
        echo "$algorithm,$gamma,$value_steps,$env,$mean_return,0.0" >> "$RESULTS_FILE"
    else
        echo "$algorithm,$gamma,$value_steps,$env,NaN,0.0" >> "$RESULTS_FILE"
    fi
}

export -f run_experiment

cleanup() {
    echo "Killing all background processes..."
    pkill -P $$
    exit 1
}

trap cleanup SIGINT SIGTERM

for env in "${ENVS[@]}"; do
    for gamma in "0.95" "1.0"; do
        # PPO
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
        #run_experiment "PPO" "$gamma" 1 "$env" &

        # VPG with value steps 1, 10, 100
        for vs in 1 10 100; do
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 1; done
            run_experiment "VPG" "$gamma" $vs "$env" &
        done
    done
done

wait

echo -e "\n=== Final Results ==="
echo "algorithm,gamma,value_steps,env,mean_return,std_return" > "results/summary_sorted.csv"
tail -n +2 "$RESULTS_FILE" | sort -t, -k4,4 -k1,1 -k2,2n -k3,3n >> "results/summary_sorted.csv"
column -t -s, "results/summary_sorted.csv"

echo -e "\nAll experiments completed! Results saved to results/summary_results.csv"
echo "Sorted results saved to results/summary_sorted.csv"
