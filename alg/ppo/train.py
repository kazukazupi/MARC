import argparse
import csv
import os
import random
from typing import Dict, List

import torch

from alg.coea.structure import BaseRobotStructure, DummyRobotStructure, Structure
from alg.ppo.env_wrappers import MultiAgentVecPytorch, make_multi_agent_vec_envs
from alg.ppo.model import Agent
from alg.ppo.ppo import PPO
from alg.ppo.ppo_utils import update_linear_schedule
from alg.ppo.storage import RolloutStorage
from utils import AGENT_1, AGENT_2, AGENT_IDS, AgentID, get_opponent_id


def train(
    args: argparse.Namespace,
    structures: Dict[AgentID, BaseRobotStructure],
):

    for s in structures.values():
        if isinstance(s, Structure):
            assert not s.is_trained, "already trained."

    agents: Dict[AgentID, Agent] = {}
    opponents: Dict[AgentID, Agent] = {}
    updaters: Dict[AgentID, PPO] = {}
    rollouts: Dict[AgentID, RolloutStorage] = {}
    train_csv_paths: Dict[AgentID, str] = {}
    vec_envs: Dict[AgentID, MultiAgentVecPytorch] = {}
    opponents_last_obs: Dict[AgentID, torch.Tensor] = {}
    controller_paths: Dict[AgentID, List[str]] = {}

    for a in AGENT_IDS:

        structure = structures[a]
        if isinstance(structure, DummyRobotStructure):
            continue
        assert isinstance(structure, Structure)

        o = get_opponent_id(a)

        # Initialize environment
        vec_envs[a] = make_multi_agent_vec_envs(
            args.env_name,
            args.num_processes,
            args.gamma,
            args.device,
            training={a: True, o: False},
            body_1=structures[AGENT_1].body,
            body_2=structures[AGENT_2].body,
            connections_1=structures[AGENT_1].connections,
            connections_2=structures[AGENT_2].connections,
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
        if isinstance(structures[o], Structure):
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

        train_csv_paths[a] = os.path.join(structure.save_path, "train_log.csv")
        controller_paths[a] = []

        with open(train_csv_paths[a], "w") as f:
            writer = csv.writer(f)
            writer.writerow(["updates", "num timesteps", "train reward"])

    for j in range(args.num_updates):

        for a in AGENT_IDS:

            structure = structures[a]
            if isinstance(structure, DummyRobotStructure):
                continue
            assert isinstance(structure, Structure)
            o = get_opponent_id(a)

            update_linear_schedule(updaters[a].optimizer, j, args.num_updates, args.lr)

            for step in range(args.num_steps):

                actions = {}
                with torch.no_grad():
                    value, actions[a], action_log_prob = agents[a].act(rollouts[a].obs[step])
                    if isinstance(structures[o], Structure):
                        _, actions[o], _ = opponents[o].act(opponents_last_obs[o], deterministic=True)

                observations, rewards, dones, infos = vec_envs[a].step(actions)

                for info in infos[a]:
                    if "episode" in info.keys():
                        with open(train_csv_paths[a], "a") as f:
                            total_num_steps = args.num_processes * (j * args.num_steps + step + 1)
                            writer = csv.writer(f)
                            writer.writerow([j, total_num_steps, info["episode"]["r"]])
                        if isinstance(structures[o], Structure) and controller_paths[o]:
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
                controller_path = os.path.join(structure.save_path, f"controller_{j}.pt")
                controller_paths[a].append(controller_path)
                torch.save(
                    [
                        agents[a].state_dict(),
                        vec_envs[a].obs_rms_dict[a],
                    ],
                    controller_path,
                )

    for a in AGENT_IDS:
        structure = structures[a]
        if isinstance(structure, DummyRobotStructure):
            continue
        assert isinstance(structure, Structure)
        structure.is_trained = True
