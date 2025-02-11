import csv
import os
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
from evogym import get_full_connectivity

from alg.ppo import PPO, Agent, RolloutStorage, update_linear_schedule
from envs import AgentID, make_vec_envs
from evaluate import evaluate
from utils import get_args


def main():

    args = get_args()
    os.mkdir(args.save_path)

    body_1 = np.array(
        [
            [0, 0, 0, 2, 3],
            [2, 0, 4, 4, 4],
            [1, 3, 2, 0, 1],
            [1, 1, 1, 3, 4],
            [0, 3, 0, 2, 0],
        ]
    )
    connections_1 = get_full_connectivity(body_1)

    body_2 = np.fliplr(body_1)
    connections_2 = get_full_connectivity(body_2)

    vec_env = make_vec_envs(
        args.env_name,
        args.num_processes,
        args.gamma,
        args.device,
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    agents: Dict[AgentID, Agent] = {}
    opponents: Dict[AgentID, Agent] = {}
    updaters: Dict[AgentID, PPO] = {}
    rollouts: Dict[AgentID, RolloutStorage] = {}
    max_determ_avg_rewards: Dict[AgentID, np.ndarray] = {}
    log_dirs: Dict[AgentID, str] = {}
    train_csv_paths: Dict[AgentID, str] = {}
    eval_csv_paths: Dict[AgentID, str] = {}
    vec_envs = {}
    opponents_last_obs = {}

    agnet_ids = ["robot_1", "robot_2"]

    for a, o in zip(agnet_ids, reversed(agnet_ids)):

        # Initialize environment
        # TODO: trainingのTrue/Falseをエージェントごとに個別設定
        vec_envs[a] = make_vec_envs(
            args.env_name,
            args.num_processes,
            args.gamma,
            args.device,
            body_1=body_1,
            body_2=body_2,
            connections_1=connections_1,
            connections_2=connections_2,
        )

        # Create agent
        obs_dim = vec_env.observation_space(a).shape[0]
        action_dim = vec_env.action_space(a).shape[0]
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

        # Create log files
        log_dirs[a] = os.path.join(args.save_path, a)
        os.mkdir(log_dirs[a])

        train_csv_paths[a] = os.path.join(log_dirs[a], "train_log.csv")
        eval_csv_paths[a] = os.path.join(log_dirs[a], "eval_log.csv")

        with open(train_csv_paths[a], "w") as f:
            writer = csv.writer(f)
            writer.writerow(["updates", "num timesteps", "train reward"])

        with open(eval_csv_paths[a], "w") as f:
            writer = csv.writer(f)
            writer.writerow(["updates", "num timesteps", "eval rewrard"])

        max_determ_avg_rewards[a] = float("-inf")

    actions = {"robot_1": None, "robot_2": None}

    for j in range(args.num_updates):

        for a, o in zip(agnet_ids, reversed(agnet_ids)):

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


if __name__ == "__main__":
    main()
