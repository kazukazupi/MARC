import argparse
import csv
import os
import random
from typing import Dict, List

import numpy as np
import torch
from evogym import get_full_connectivity  # type: ignore

from alg.coea.structure import Structure
from alg.ppo.env_wrappers import make_multi_agent_vec_envs, make_single_agent_vec_env
from alg.ppo.model import Agent
from alg.ppo.ppo import PPO
from alg.ppo.ppo_utils import update_linear_schedule
from alg.ppo.storage import RolloutStorage
from envs import AgentID
from utils import get_agent_names


def train(
    args: argparse.Namespace,
    structures: Dict[AgentID, Structure],
):

    assert any([not s.is_trained for s in structures.values()]), "already trained."

    agents: Dict[AgentID, Agent] = {}
    opponents: Dict[AgentID, Agent] = {}
    updaters: Dict[AgentID, PPO] = {}
    rollouts: Dict[AgentID, RolloutStorage] = {}
    train_csv_paths: Dict[AgentID, str] = {}
    vec_envs = {}
    opponents_last_obs: Dict[AgentID, torch.Tensor] = {}
    controller_paths: Dict[AgentID, List[str]] = {}

    agent_names = get_agent_names()

    for a, o in zip(agent_names, reversed(agent_names)):

        # Initialize environment
        vec_envs[a] = make_multi_agent_vec_envs(
            args.env_name,
            args.num_processes,
            args.gamma,
            args.device,
            training={a: True, o: False},
            body_1=structures[agent_names[0]].body,
            body_2=structures[agent_names[1]].body,
            connections_1=structures[agent_names[0]].connections,
            connections_2=structures[agent_names[1]].connections,
        )

        # Create agent
        obs_dim = vec_envs[a].observation_space(a).shape[0]
        action_dim = vec_envs[a].action_space(a).shape[0]
        agents[a] = Agent(
            obs_dim=obs_dim,
            hidden_dim=64,
            action_dim=action_dim,
        )
        agents[a].to(args.device)

        # Create opponent
        opponents[a] = Agent(
            obs_dim=obs_dim,
            hidden_dim=64,
            action_dim=action_dim,
        )
        opponents[a].to(args.device)
        opponents[a].load_state_dict(agents[a].state_dict())

        # Create updater
        updaters[a] = PPO(
            agents[a],
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            args.lr,
            args.eps,
            args.max_grad_norm,
        )

        # Create rollouts
        observations = vec_envs[a].reset()
        rollouts[a] = RolloutStorage(
            num_steps=args.num_steps,
            num_processes=args.num_processes,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        rollouts[a].obs[0].copy_(observations[a])
        rollouts[a].to(args.device)
        opponents_last_obs[o] = observations[o]

        train_csv_paths[a] = os.path.join(structures[a].save_path, "train_log.csv")
        controller_paths[a] = []

        with open(train_csv_paths[a], "w") as f:
            writer = csv.writer(f)
            writer.writerow(["updates", "num timesteps", "train reward"])

    actions = {}

    for j in range(args.num_updates):

        for a, o in zip(agent_names, reversed(agent_names)):

            update_linear_schedule(updaters[a].optimizer, j, args.num_updates, args.lr)

            for step in range(args.num_steps):

                with torch.no_grad():
                    value, actions[a], action_log_prob = agents[a].act(rollouts[a].obs[step])
                    _, actions[o], _ = opponents[o].act(opponents_last_obs[o], deterministic=True)

                observations, rewards, dones, infos = vec_envs[a].step(actions)

                for info in infos[a]:
                    if "episode" in info.keys():
                        with open(train_csv_paths[a], "a") as f:
                            total_num_steps = args.num_processes * (j * args.num_steps + step + 1)
                            writer = csv.writer(f)
                            writer.writerow([j, total_num_steps, info["episode"]["r"]])
                        if controller_paths[o]:
                            delta_index = int(args.delta * len(controller_paths[o]))
                            recent_controllers = controller_paths[o][delta_index:]
                            if recent_controllers:
                                controller_path = random.choice(recent_controllers)
                            else:
                                controller_path = controller_paths[o][-1]
                            state_dict, obs_rms = torch.load(controller_path)
                            opponents[o].load_state_dict(state_dict)
                            vec_envs[a].obs_rms_dict[o] = obs_rms

                masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones[a]])
                bad_masks = torch.FloatTensor([[0.0] if info["TimeLimit.truncated"] else [1.0] for info in infos[a]])

                rollouts[a].insert(observations[a], actions[a], action_log_prob, value, rewards[a], masks, bad_masks)
                opponents_last_obs[o] = observations[o]

            with torch.no_grad():
                next_value = agents[a].get_value(rollouts[a].obs[-1]).detach()

            rollouts[a].compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits
            )
            updaters[a].update(rollouts[a])
            rollouts[a].after_update()

            if j % args.save_interval == 0:
                controller_path = os.path.join(structures[a].save_path, f"controller_{j}.pt")
                controller_paths[a].append(controller_path)
                torch.save(
                    [
                        agents[a].state_dict(),
                        vec_envs[a].obs_rms_dict[a],
                    ],
                    controller_path,
                )

    for a in agent_names:
        structures[a].is_trained = True


def train_against_fixed_opponent(
    args: argparse.Namespace,
    self_structure: Structure,
    self_agent_name: AgentID,
    opponent_agent_name: AgentID,
):

    assert not self_structure.is_trained, "already trained."

    opponent_body = np.ones_like(self_structure.body) * 2
    opponent_connections = get_full_connectivity(opponent_body)

    # Initialize environment
    vec_env = make_single_agent_vec_env(
        args.env_name,
        args.num_processes,
        args.gamma,
        args.device,
        self_id=self_agent_name,
        opponent_id=opponent_agent_name,
        body_1=self_structure.body,
        body_2=opponent_body,
        connections_1=self_structure.connections,
        connections_2=opponent_connections,
    )

    # Create agent
    obs_dim = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_space.shape[0]
    agent = Agent(
        obs_dim=obs_dim,
        hidden_dim=64,
        action_dim=action_dim,
    )
    agent.to(args.device)

    # Create updater
    updater = PPO(
        agent,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.lr,
        args.eps,
        args.max_grad_norm,
    )

    # Create rollouts
    observations = vec_env.reset()
    rollouts = RolloutStorage(
        num_steps=args.num_steps,
        num_processes=args.num_processes,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    rollouts.obs[0].copy_(observations)
    rollouts.to(args.device)

    # Prepare for logging
    train_csv_path = os.path.join(self_structure.save_path, "train_log.csv")
    with open(train_csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["updates", "num timesteps", "train reward"])

    for j in range(args.num_updates):

        update_linear_schedule(updater.optimizer, j, args.num_updates, args.lr)

        for step in range(args.num_steps):

            with torch.no_grad():
                value, action, action_log_prob = agent.act(rollouts.obs[step])

            observations, rewards, dones, infos = vec_env.step(action)

            for info in infos:
                if "episode" in info.keys():
                    with open(train_csv_path, "a") as f:
                        total_num_steps = args.num_processes * (j * args.num_steps + step + 1)
                        writer = csv.writer(f)
                        writer.writerow([j, total_num_steps, info["episode"]["r"]])

            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])
            bad_masks = torch.FloatTensor([[0.0] if info["TimeLimit.truncated"] else [1.0] for info in infos])

            rollouts.insert(observations, action, action_log_prob, value, rewards, masks, bad_masks)

        with torch.no_grad():
            next_value = agent.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        updater.update(rollouts)
        rollouts.after_update()

        if j % args.save_interval == 0:
            controller_path = os.path.join(self_structure.save_path, f"controller_{j}.pt")
            torch.save(
                [
                    agent.state_dict(),
                    vec_env.obs_rms,
                ],
                controller_path,
            )

    self_structure.is_trained = True
