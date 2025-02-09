import csv
import os
from collections import deque

import numpy as np
import torch
from evogym import get_full_connectivity

from alg.ppo import PPO, Agent, RolloutStorage, update_linear_schedule
from envs import make_vec_envs
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
    body_2 = np.array(
        [
            [0, 0, 3, 3, 3],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0],
        ]
    )
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

    obs_dim = vec_env.observation_space("robot_1").shape[0]
    action_dim = vec_env.action_space("robot_1").shape[0]

    agent = Agent(
        obs_dim=obs_dim,
        hidden_dim=64,
        action_dim=action_dim,
    )
    agent.to(args.device)

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

    rollouts = RolloutStorage(
        num_steps=args.num_steps,
        num_processes=args.num_processes,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    observations = vec_env.reset()
    obs = observations["robot_1"]
    rollouts.obs[0].copy_(obs)
    rollouts.to(args.device)

    episode_rewards = deque(maxlen=10)

    max_determ_avg_reward = float("-inf")

    train_csv_path = os.path.join(args.save_path, "train_log.csv")
    eval_csv_path = os.path.join(args.save_path, "eval_log.csv")

    with open(train_csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["updates", "num timesteps", "mean reward", "median reward", "min reward", "max reward"])

    with open(eval_csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["updates", "num timesteps", "rewrard", "max reward"])

    for j in range(args.num_updates):

        update_linear_schedule(updater.optimizer, j, args.num_updates, args.lr)

        for step in range(args.num_steps):

            with torch.no_grad():
                value, action, action_log_prob = agent.act(rollouts.obs[step])

            action_2 = torch.Tensor(np.array([vec_env.action_space("robot_2").sample()]))

            actions = {"robot_1": action, "robot_2": action_2}

            observations, rewards_, dones_, infos_ = vec_env.step(actions)
            obs = observations["robot_1"]
            rewards = rewards_["robot_1"]
            dones = dones_["robot_1"]
            infos = infos_["robot_1"]

            for info in infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])

            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])
            bad_masks = torch.FloatTensor([[0.0] if info["TimeLimit.truncated"] else [1.0] for info in infos])

            rollouts.insert(obs, action, action_log_prob, value, rewards, masks, bad_masks)

        with torch.no_grad():
            next_value = agent.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        updater.update(rollouts)

        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            with open(train_csv_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        j,
                        total_num_steps,
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                    ]
                )

            print(
                "Updates {}, num timesteps {}\nLast {} training episodes:"
                "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                )
            )

        if j % args.eval_interval == 0 and len(episode_rewards) > 1:

            obs_rms_dict = vec_env.obs_rms_dict
            results = evaluate(
                [agent, None],
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

            determ_avg_reward = results["robot_1"]

            print(f"Evaluated using {args.num_evals} episodes. Mean reward: {determ_avg_reward}\n")

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward
                controller_path = os.path.join(args.save_path, "agent1_controller.pt")
                print(f"Saving {controller_path} with avg reward {max_determ_avg_reward}\n")
                torch.save(
                    [
                        agent.state_dict(),
                        obs_rms_dict["robot_1"],
                    ],
                    controller_path,
                )

            with open(eval_csv_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([j, total_num_steps, determ_avg_reward, max_determ_avg_reward])

    results = evaluate(
        [agent, None],
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
