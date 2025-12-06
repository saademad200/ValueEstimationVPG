# This implementation is adapted and modified from https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "vpg-mujoco"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""


    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 7e-4
    """the learning rate of the optimizer for the actor"""
    critic_learning_rate: float = 3e-4
    """the learning rate of the optimizer for the critic"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_value_step: int = 50
    """the lnumber of value steps per iteration"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
   #     env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
    #    env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
    #    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
     #   env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_mean(self, x):
        return self.actor_mean(x)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__VPG_vs{args.num_value_step}_g{args.gamma}_s{args.seed}"

    # Initialize wandb if tracking is enabled
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Initialize TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.make_vec(args.env_id, num_envs=args.num_envs, vectorization_mode="async")
    envs2 = gym.make_vec(args.env_id, num_envs=1, vectorization_mode="async")
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

  #  envs = gym.wrappers.FlattenObservation(envs)  # deal with dm_control's Dict observation space
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

  #  envs = gym.wrappers.vector.ClipAction(envs)
#    envs = gym.wrappers.vector.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.vector.NormalizeObservation(envs)
#   envs2 = gym.wrappers.vector.ClipAction(envs2)
    envs2 = gym.wrappers.vector.NormalizeObservation(envs2)

    agent = Agent(envs).to(device)
    actor_optimizer = optim.Adam(agent.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = optim.Adam(agent.parameters(), lr=args.critic_learning_rate)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    reward_curve = []
    for iteration in range(1, args.num_iterations + 1):

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
        # Value loss
        for i in range(args.num_value_step):
            newvalue = agent.get_value(b_obs)
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
            critic_optimizer.zero_grad()
            v_loss.backward()
            critic_optimizer.step()

        # Policy loss
        pg_loss = (-b_advantages * newlogprob).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss
        actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        actor_optimizer.step()

        # Log metrics for this iteration
        episode_rewards = []
        for i, info in enumerate(infos):
            if isinstance(info, dict) and 'episode' in info:
                ep_ret = info['episode']['r']
                ep_len = info['episode']['l']
                episode_rewards.append(ep_ret)
                writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                writer.add_scalar("charts/episodic_length", ep_len, global_step)

        # Average episodic reward for this iteration
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            writer.add_scalar("charts/avg_episode_reward", avg_reward, global_step)
            # Also log as train/reward to match PPO (Tianshou naming)
            writer.add_scalar("train/reward", avg_reward, global_step)
            writer.add_scalar("train/length", np.mean([info['episode']['l'] for info in infos if isinstance(info, dict) and 'episode' in info]), global_step)
            print(f"Iteration {iteration}, Step {global_step}, Avg Reward: {avg_reward}")

        # Cumulative reward across all steps of this iteration
        cum_reward = rewards.sum().item()
        writer.add_scalar("charts/cumulative_reward", cum_reward, global_step)

        # Log training metrics (use global_step for consistency with PPO)
        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


        # evaluate policy every 10240 steps (matches PPO step-per-epoch for aligned charts)
        if (((global_step + args.batch_size) // 10240) - ((global_step) // 10240)) > 0:
            envs2.obs_rms = envs.obs_rms
            obser, info = envs2.reset(seed=args.seed)
            S = 0.
            for k in range(1000):
                state = torch.tensor(obser, dtype=torch.float).to(device)
                with torch.no_grad():
                    v = agent.get_mean(state)
                obser, reward, terminated, truncated, info = envs2.step(v.cpu().numpy())
                S = S + reward
                if terminated or truncated:
                    break
            # Log as test/reward to match PPO (Tianshou naming)
            writer.add_scalar("test/reward", S[0], global_step)
            print("Eval (test) Reward:", S[0])
            print("Iteration", iteration)
            reward_curve.append((global_step, S[0]))  # Store (timestep, reward) tuple

    envs.close()
    print("Time elapsed (s):", time.time() - start_time)
    
    # Save learning curve to CSV for plotting
    import csv
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/learning_curve_vpg_{args.env_id}_s{args.seed}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['timestep', 'reward', 'algorithm', 'env', 'seed'])
        for timestep, reward in reward_curve:
            writer_csv.writerow([timestep, reward, 'VPG', args.env_id, args.seed])
    print(f"Saved learning curve to {csv_path}")

