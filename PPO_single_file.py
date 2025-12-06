#!/usr/bin/env python3
"""
CleanRL style PPO for MuJoCo environments.
"""

import os
import time
import random
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import tyro


@dataclass
class Args:
    exp_name: str = "ppo_mujoco"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "mujoco"
    wandb_entity: str = ""
    capture_video: bool = False

    env_id: str = "Hopper-v4"
    num_envs: int = 64

    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 64
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    dual_clip: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    eval_interval: int = 10000


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)

        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        return action, logprob, entropy, self.critic(x)


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return env
    return thunk


def main(args: Args):
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity or None,
            sync_tensorboard=True,
            config=vars(args),
            name=args.exp_name,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)

    if args.num_iterations < 1:
        raise ValueError("Calculated zero iterations. Increase total_timesteps or reduce num_envs*num_steps")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
                                     for i in range(args.num_envs)])

    eval_env = gym.make(args.env_id)
    eval_env = gym.wrappers.NormalizeObservation(eval_env)
    eval_env = gym.wrappers.NormalizeReward(eval_env, gamma=args.gamma)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()
    next_eval_step = args.eval_interval

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        ep_returns_this_iter = []

        # rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()

            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(action.cpu().numpy())
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(np.logical_or(term_np, trunc_np), dtype=torch.float32, device=device)

            for info in infos:
                if isinstance(info, dict) and "episode" in info:
                    ep_returns_this_iter.append(info["episode"]["r"])

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value[0]
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds].to(device),
                    b_actions[mb_inds].to(device),
                )

                logratio = newlogprob - b_logprobs[mb_inds].to(device)
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = torch.mean((ratio - 1) - logratio)
                    clipfracs.append(((ratio - 1).abs() > args.clip_coef).float().mean())

                mb_adv = b_advantages[mb_inds].to(device)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb_inds].to(device)) ** 2
                    v_clipped = b_values[mb_inds].to(device) + torch.clamp(
                        newvalue - b_values[mb_inds].to(device),
                        -args.clip_coef,
                        args.clip_coef
                    )
                    v_clipped = (v_clipped - b_returns[mb_inds].to(device)) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds].to(device)) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss - args.ent_coef * ent + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                last_pg = pg_loss.item()
                last_v = v_loss.item()
                last_ent = ent.item()

            if args.target_kl and approx_kl > args.target_kl:
                break

        avg_reward = float(np.mean(ep_returns_this_iter)) if ep_returns_this_iter else 0.0
        writer.add_scalar("charts/avg_reward", avg_reward, iteration)
        writer.add_scalar("losses/value", last_v, iteration)
        writer.add_scalar("losses/policy", last_pg, iteration)
        writer.add_scalar("losses/entropy", last_ent, iteration)

        if args.track:
            import wandb
            wandb.log({
                "avg_reward": avg_reward,
                "value_loss": last_v,
                "policy_loss": last_pg,
                "entropy": last_ent,
                "clipfrac": np.mean(clipfracs) if clipfracs else 0,
            }, step=global_step)

        # evaluation
        if global_step >= next_eval_step:
            next_eval_step += args.eval_interval
            eval_obs, _ = eval_env.reset()
            eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)
            done = False
            total_eval = 0.0

            while not done:
                with torch.no_grad():
                    action = agent.actor_mean(eval_obs)
                obs_raw, reward, term, trunc, _ = eval_env.step(action.cpu().numpy()[0])
                total_eval += float(reward)
                done = bool(term) or bool(trunc)
                eval_obs = torch.tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)

            writer.add_scalar("eval/return", total_eval, global_step)
            if args.track:
                wandb.log({"eval_return": total_eval}, step=global_step)

    envs.close()
    eval_env.close()
    writer.close()
    if args.track:
        import wandb
        wandb.finish()

    print("Training done. Total time:", time.time() - start_time)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
