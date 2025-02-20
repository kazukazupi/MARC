from collections import deque

import numpy as np
import torch
from evogym import envs, get_full_connectivity  # noqa

from alg.ppo import PPO, Agent, RolloutStorage, evaluate, make_vec_envs, update_linear_schedule


def main():

    gamma = 0.99
    # num_env_steps = int(10e6)
    # num_updates = 1000
    num_updates = 300
    num_processes = 4
    num_steps = 128
    gae_lambda = 0.95
    env_name = "Walker-v0"
    clip_param = 0.1
    ppo_epoch = 4
    num_mini_batch = 4
    value_loss_coef = 0.5
    entropy_coef = 0.01
    lr = 2.5e-4
    eps = 1e-5
    max_grad_norm = 0.5
    use_gae = True
    use_proper_time_limits = False
    seed = 42

    body = np.array(
        [
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
            [3, 3, 0, 3, 3],
            [3, 3, 0, 3, 3],
        ]
    )
    connections = get_full_connectivity(body)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_vec_envs(
        env_name=env_name,
        num_processes=num_processes,
        gamma=gamma,
        device=device,
        training=True,
        seed=seed,
        body=body,
        connections=connections,
    )

    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    agent = Agent(
        obs_dim=obs_dim,
        hidden_dim=64,
        action_dim=action_dim,
    )
    agent.to(device)

    updater = PPO(agent, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef, lr, eps, max_grad_norm)

    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    obs = env.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards: deque[float] = deque(maxlen=10)

    # num_updates = num_env_steps // num_steps // num_processes

    for j in range(num_updates):

        update_linear_schedule(updater.optimizer, j, num_updates, lr)

        # Sample Actions
        for step in range(num_steps):

            with torch.no_grad():
                value, action, action_log_prob = agent.act(rollouts.obs[step])

            obs, rewards, dones, infos = env.step(action)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])
            bad_masks = torch.FloatTensor([[0.0] if info["TimeLimit.truncated"] else [1.0] for info in infos])

            rollouts.insert(obs, action, action_log_prob, value, rewards, masks, bad_masks)

        with torch.no_grad():
            next_value = agent.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)

        updater.update(rollouts)

        rollouts.after_update()

        if j % 1 == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_processes * num_steps

            print(
                "Updates {}, num timesteps {}\nLast {} training episodes:"
                "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    len(episode_rewards),
                    float(np.mean(episode_rewards)),
                    float(np.median(episode_rewards)),
                    float(np.min(episode_rewards)),
                    float(np.max(episode_rewards)),
                )
            )

    me = evaluate(
        agent,
        env.obs_rms,
        env_name,
        num_processes=1,
        device=device,
        min_num_episodes=1,
        seed=seed,
        body=body,
        connections=connections,
        render_mode="human",
    )
    print(f"Mean Episode Reward: {me}")


if __name__ == "__main__":
    main()
