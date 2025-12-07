# VPG-Hybrid Experiment: The Ultimate Combination
# This experiment combines the "Best of Class" features:
# 1. Normalized GAE (Stability)
# 2. Large Critic Network (Accuracy)
# 3. PPO-style Clipping (Safety)
# 4. Adaptive Value Steps (Efficiency)
#
# Hypothesis: This hybrid approach will offer the best trade-off between performance, stability, and training time.

import os
import random
import time
from dataclasses import dataclass, field
from typing import List
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
    exp_name: str = "vpg_hybrid"
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
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    
    # Feature 4: Adaptive Value Steps Parameters
    min_value_steps: int = 10
    """minimum number of value steps"""
    max_value_steps: int = 100
    """maximum number of value steps"""
    value_loss_high: float = 0.5
    """value loss threshold to increase steps"""
    value_loss_low: float = 0.1
    """value loss threshold to decrease steps"""
    
    # Feature 1: Normalized GAE
    norm_adv: bool = True
    """if toggled, advantages will be normalized"""

    # Feature 2: Large Critic (defaults to 128x128)
    actor_sizes: List[int] = field(default_factory=lambda: [64, 64])
    """hidden layer sizes for actor network"""
    critic_sizes: List[int] = field(default_factory=lambda: [128, 128])
    """hidden layer sizes for critic network"""

    # Feature 3: PPO Clipping
    clip_eps: float = 0.2
    """PPO clipping epsilon"""
    
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


def build_mlp(input_dim, hidden_sizes, output_dim, output_std=1.0):
    """Build MLP with configurable hidden sizes."""
    layers = []
    prev_dim = input_dim
    for size in hidden_sizes:
        layers.append(layer_init(nn.Linear(prev_dim, size)))
        layers.append(nn.Tanh())
        prev_dim = size
    layers.append(layer_init(nn.Linear(prev_dim, output_dim), std=output_std))
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, envs, actor_sizes, critic_sizes):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        
        # Build critic with configurable size
        self.critic = build_mlp(obs_dim, critic_sizes, 1, output_std=1.0)
        
        # Build actor with configurable size
        self.actor_mean = build_mlp(obs_dim, actor_sizes, action_dim, output_std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Log architecture info
        critic_params = sum(p.numel() for p in self.critic.parameters())
        actor_params = sum(p.numel() for p in self.actor_mean.parameters())
        print(f"Critic architecture: {critic_sizes}, params: {critic_params}")
        print(f"Actor architecture: {actor_sizes}, params: {actor_params}")

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


def get_adaptive_value_steps(v_loss, args, current_steps):
    """Adaptively adjust value steps based on value loss."""
    if v_loss > args.value_loss_high:
        # Value function struggling, use more steps
        return min(current_steps + 10, args.max_value_steps)
    elif v_loss < args.value_loss_low:
        # Value function accurate, use fewer steps
        return max(current_steps - 10, args.min_value_steps)
    else:
        return current_steps


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.env_id}__VPG_HYBRID_s{args.seed}"

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

    agent = Agent(envs, list(args.actor_sizes), list(args.critic_sizes)).to(device)
    actor_optimizer = optim.Adam(agent.actor_mean.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.critic_learning_rate)

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

    # Adaptive value steps tracking
    current_value_steps = (args.min_value_steps + args.max_value_steps) // 2
    total_value_steps_used = 0

    # Track clipping statistics
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

        # Bootstrap value
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

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Feature 1: Normalize advantages (like PPO)
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Feature 4: Value loss with ADAPTIVE steps
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
        
        for i in range(current_value_steps):
            newvalue = agent.get_value(b_obs)
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
            critic_optimizer.zero_grad()
            v_loss.backward()
            critic_optimizer.step()
        
        total_value_steps_used += current_value_steps
        
        # Adapt value steps for next iteration
        current_value_steps = get_adaptive_value_steps(v_loss.item(), args, current_value_steps)

        # Feature 3: PPO-style clipped policy loss
        _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs, b_actions)
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()
        
        # Calculate clipping fraction
        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > args.clip_eps).float().mean().item()
            clip_fractions.append(clip_frac)
        
        # Clipped surrogate objective
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
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
            print(f"Step {global_step}, Reward: {avg_reward:.2f}, Clip: {clip_frac:.3f}, VS: {current_value_steps}")

        # Cumulative reward across all steps of this iteration
        cum_reward = rewards.sum().item()
        writer.add_scalar("charts/cumulative_reward", cum_reward, global_step)

        writer.add_scalar("charts/clip_fraction", clip_frac, global_step)
        writer.add_scalar("charts/value_steps", current_value_steps, global_step)
        writer.add_scalar("charts/cumulative_value_steps", total_value_steps_used, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/time_elapsed", time.time() - start_time, global_step)

        # Evaluate every 10240 steps (matches PPO)
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
    print(f"Average clip fraction: {np.mean(clip_fractions):.3f}")
    avg_value_steps = total_value_steps_used / args.num_iterations
    print(f"Average value steps per iteration: {avg_value_steps:.1f}")

    # Save learning curve to CSV for plotting
    import csv
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/learning_curve_{args.exp_name}_{args.env_id}_s{args.seed}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['timestep', 'reward', 'algorithm', 'env', 'seed'])
        for timestep, reward in reward_curve:
            writer_csv.writerow([timestep, reward, 'VPG', args.env_id, args.seed])
    print(f"Saved learning curve to {csv_path}")
