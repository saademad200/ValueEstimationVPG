# VPG Combo: Flexible Experimentation
# Supports combinations of:
# - Advantage Normalization
# - Adaptive Value Steps
# - PPO-style Clipping
# - Monte Carlo vs GAE

import os
import random
import time
from dataclasses import dataclass
from typing import Literal
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
    exp_name: str = "vpg_combo"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "vpg-experiments"
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
    
    # Advantage Params
    advantage_type: Literal["gae", "mc"] = "gae"
    """advantage estimation method: 'gae' or 'mc'"""
    gae_lambda: float = 0.95
    """the lambda for GAE"""
    norm_adv: bool = False
    """if toggled, advantages will be normalized"""

    # Value Step Params
    num_value_step: int = 50
    """the number of fixed value steps per iteration"""
    adaptive_value_steps: bool = False
    """if toggled, use adaptive value steps instead of fixed"""
    min_value_steps: int = 10
    max_value_steps: int = 100
    value_loss_high: float = 0.5
    value_loss_low: float = 0.1

    # Clipping Params
    clip_eps: float = 0.0
    """PPO clipping epsilon (if > 0.0, enables clipping)"""

    ent_coef: float = 0.0
    """coefficient of the entropy"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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


def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda, device):
    """Compute advantages using Generalized Advantage Estimation (GAE)."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


def compute_mc_returns(rewards, dones, next_value, next_done, gamma, device):
    """Compute Monte Carlo returns (full discounted sum, no bootstrapping)."""
    num_steps = rewards.shape[0]
    returns = torch.zeros_like(rewards).to(device)
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            next_return = next_value * nextnonterminal  # Use V for incomplete episodes
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            next_return = returns[t + 1] * nextnonterminal
        returns[t] = rewards[t] + gamma * next_return
    
    return returns


def get_adaptive_value_steps(v_loss, args, current_steps):
    """Adaptively adjust value steps based on value loss."""
    if v_loss > args.value_loss_high:
        return min(current_steps + 10, args.max_value_steps)
    elif v_loss < args.value_loss_low:
        return max(current_steps - 10, args.min_value_steps)
    else:
        return current_steps


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # Construct descriptive run name based on flags
    name_parts = [args.env_id, "VPG"]
    if args.advantage_type == "mc":
        name_parts.append("MC")
    elif args.advantage_type == "gae":
        name_parts.append(f"GAE{args.gae_lambda}")
        
    if args.norm_adv:
        name_parts.append("Norm")
    
    if args.clip_eps > 0:
        name_parts.append(f"Clip{args.clip_eps}")
        
    if args.adaptive_value_steps:
        name_parts.append("AdaptiveVS")
    else:
        name_parts.append(f"vs{args.num_value_step}")
        
    run_name = "__".join(name_parts) + f"_s{args.seed}"

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

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.make_vec(args.env_id, num_envs=args.num_envs, vectorization_mode="async")
    envs2 = gym.make_vec(args.env_id, num_envs=1, vectorization_mode="async")
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.vector.NormalizeObservation(envs)
    envs2 = gym.wrappers.vector.NormalizeObservation(envs2)

    agent = Agent(envs).to(device)
    actor_optimizer = optim.Adam(agent.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = optim.Adam(agent.parameters(), lr=args.critic_learning_rate)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Value steps state
    current_value_steps = args.num_value_step
    if args.adaptive_value_steps:
        current_value_steps = (args.min_value_steps + args.max_value_steps) // 2
    
    total_value_steps_used = 0
    clip_fractions = []

    reward_curve = []
    
    for iteration in range(1, args.num_iterations + 1):

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        # Advantage Calculation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            
            if args.advantage_type == "gae":
                advantages, returns = compute_gae(
                    rewards, values, dones, next_value, next_done, 
                    args.gamma, args.gae_lambda, device
                )
            elif args.advantage_type == "mc":
                returns = compute_mc_returns(
                    rewards, dones, next_value, next_done, args.gamma, device
                )
                advantages = returns - values
            else:
                 raise ValueError(f"Unknown advantage type: {args.advantage_type}")

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Normalize advantages
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Value Update
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
        
        for i in range(current_value_steps):
            newvalue = agent.get_value(b_obs)
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
            critic_optimizer.zero_grad()
            v_loss.backward()
            critic_optimizer.step()
        
        total_value_steps_used += current_value_steps
        
        # Adaptive adjustment
        if args.adaptive_value_steps:
            current_value_steps = get_adaptive_value_steps(v_loss.item(), args, current_value_steps)

        # Policy Update
        if args.clip_eps > 0.0:
            # PPO-style clipped loss
            _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs, b_actions)
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()
            
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > args.clip_eps).float().mean().item()
                clip_fractions.append(clip_frac)
            
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        else:
            # VPG-style loss
            _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs, b_actions)
            # Standard VPG: log_prob * A. 
            # Note: For on-policy VPG, logratio is 0, ratio is 1. newlogprob is recalculated.
            # Using standard implementation:
            pg_loss = (-b_advantages * newlogprob).mean()
            clip_frac = 0.0

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss
        
        actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        actor_optimizer.step()

        # Logging
        episode_rewards = []
        for i, info in enumerate(infos):
            if isinstance(info, dict) and 'episode' in info:
                ep_ret = info['episode']['r']
                episode_rewards.append(ep_ret)
                writer.add_scalar("charts/episodic_return", ep_ret, global_step)

        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            writer.add_scalar("charts/avg_episode_reward", avg_reward, global_step)
            writer.add_scalar("train/reward", avg_reward, global_step)
            print(f"Step {global_step}, Reward: {avg_reward:.2f}, VS: {current_value_steps}")

        cum_reward = rewards.sum().item()
        writer.add_scalar("charts/cumulative_reward", cum_reward, global_step)
        
        if args.clip_eps > 0.0:
            writer.add_scalar("charts/clip_fraction", clip_frac, global_step)
            
        writer.add_scalar("charts/value_steps", current_value_steps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/time_elapsed", time.time() - start_time, global_step)

        # Evaluation
        if (((global_step + args.batch_size) // 10240) - ((global_step) // 10240)) > 0:
            envs2.obs_rms = envs.obs_rms
            eval_rewards = []
            for eval_ep in range(10):
                obser, info = envs2.reset(seed=int(args.seed) + eval_ep)
                ep_ret = 0
                while True:
                    state = torch.tensor(obser, dtype=torch.float).to(device)
                    with torch.no_grad():
                        action = agent.get_mean(state)
                    obser, reward, terminated, truncated, info = envs2.step(action.cpu().numpy())
                    ep_ret += reward[0]
                    if terminated[0] or truncated[0]:
                        break
                eval_rewards.append(ep_ret)
            
            avg_eval_reward = np.mean(eval_rewards)
            writer.add_scalar("test/reward", avg_eval_reward, global_step)
            print(f"Eval Reward: {avg_eval_reward:.2f}")
            reward_curve.append((global_step, avg_eval_reward))

    envs.close()
    envs2.close()
    print(f"Time elapsed: {time.time() - start_time:.2f}s")
    
    # Save learning curve
    import csv
    os.makedirs("results", exist_ok=True)
    # Use a simpler filename for the CSV to match expected patterns or make it unique
    # We will stick to the run_name pattern
    csv_path = f"results/learning_curve_{run_name}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['timestep', 'reward', 'algorithm', 'env', 'seed'])
        for timestep, reward in reward_curve:
            # We can use exp_name as algorithm identifier
            writer_csv.writerow([timestep, reward, 'VPG', args.env_id, args.seed])
    print(f"Saved learning curve to {csv_path}")
