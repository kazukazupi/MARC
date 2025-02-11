import csv
import os
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
    updaters: Dict[AgentID, PPO] = {}
    rollouts: Dict[AgentID, RolloutStorage] = {}
    max_determ_avg_rewards: Dict[AgentID, np.ndarray] = {}
    log_dirs: Dict[AgentID, str] = {}
    train_csv_paths: Dict[AgentID, str] = {}
    eval_csv_paths: Dict[AgentID, str] = {}

    observations = vec_env.reset()

    for a in vec_env.agents:

        # Create agent
        obs_dim = vec_env.observation_space(a).shape[0]
        action_dim = vec_env.action_space(a).shape[0]
        agents[a] = Agent(
            obs_dim=obs_dim,
            hidden_dim=64,
            action_dim=action_dim,
        )
        agents[a].to(args.device)

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
        rollouts[a] = RolloutStorage(
            num_steps=args.num_steps,
            num_processes=args.num_processes,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        rollouts[a].obs[0].copy_(observations[a])
        rollouts[a].to(args.device)

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

    for j in range(args.num_updates):

        for a in vec_env.agents:
            update_linear_schedule(updaters[a].optimizer, j, args.num_updates, args.lr)

        # Sample actions
        for step in range(args.num_steps):

            values = {}
            actions = {}
            action_log_probs = {}

            for a in vec_env.agents:
                with torch.no_grad():
                    values[a], actions[a], action_log_probs[a] = agents[a].act(rollouts[a].obs[step])

            observations, rewards, dones, infos = vec_env.step(actions)

            for a in vec_env.agents:

                # Store data
                masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones[a]])
                bad_masks = torch.FloatTensor([[0.0] if info["TimeLimit.truncated"] else [1.0] for info in infos[a]])
                rollouts[a].insert(
                    observations[a], actions[a], action_log_probs[a], values[a], rewards[a], masks, bad_masks
                )

                # Log
                for info in infos[a]:
                    if "episode" in info:
                        total_num_steps = (j + 1) * args.num_processes * args.num_steps
                        with open(train_csv_paths[a], "a") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    j,
                                    total_num_steps,
                                    info["episode"]["r"],
                                ]
                            )

        # Update
        next_values = {}

        for a in vec_env.agents:

            with torch.no_grad():
                next_values[a] = agents[a].get_value(rollouts[a].obs[-1]).detach()

            rollouts[a].compute_returns(
                next_values[a], args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits
            )
            updaters[a].update(rollouts[a])
            rollouts[a].after_update()

        if j % args.eval_interval == 0:

            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            obs_rms_dict = vec_env.obs_rms_dict
            determ_avg_rewards = evaluate(
                list(agents.values()),
                obs_rms_dict,
                args.env_name,
                args.num_processes,
                args.device,
                min_num_episodes=args.num_evals,
                seed=None,
                body_1=body_1,
                body_2=body_2,
                connections_1=connections_1,
                connections_2=connections_2,
            )

            for a in vec_env.agents:
                if determ_avg_rewards[a] > max_determ_avg_rewards[a]:
                    max_determ_avg_rewards[a] = determ_avg_rewards[a]
                    controller_path = os.path.join(log_dirs[a], "controller.pt")
                    torch.save(
                        [
                            agents[a].state_dict(),
                            obs_rms_dict[a],
                        ],
                        controller_path,
                    )

                with open(eval_csv_paths[a], "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([j, total_num_steps, determ_avg_rewards[a]])

    results = evaluate(
        list(agents.values()),
        vec_env.obs_rms_dict,
        args.env_name,
        args.num_processes,
        args.device,
        min_num_episodes=1,
        seed=None,
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
        render_mode="human",
    )

    print(results)


if __name__ == "__main__":
    main()
