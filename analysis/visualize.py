import argparse
import os
from typing import Any, Dict, Generator, List, Optional

import cv2  # type: ignore
import numpy as np
import torch

from alg.coea.structure import Structure
from alg.ppo import Agent, make_vec_envs
from analysis.analysis_utils import get_robot_save_path, get_top_robot_ids
from envs import AgentID
from utils import get_agent_names, load_args


def visualize(
    structures: Dict[AgentID, Structure],
    env_name: str,
    num_processes: int,
    device: torch.device,
    min_num_episodes: int = 1,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    render_options: Optional[Dict[str, Any]] = None,
) -> Generator[np.ndarray, None, None]:

    agent_names = get_agent_names()

    envs = make_vec_envs(
        env_name,
        num_processes,
        None,
        device,
        training=False,
        seed=seed,
        body_1=structures[agent_names[0]].body,
        body_2=structures[agent_names[1]].body,
        connections_1=structures[agent_names[0]].connections,
        connections_2=structures[agent_names[1]].connections,
        render_mode=render_mode,
        render_options=render_options,
    )

    agents = {}
    for a, structure in structures.items():
        controller_path = structure.get_latest_controller_path()
        state_dict, obs_rms = torch.load(controller_path, map_location=device)
        agents[a] = Agent.from_state_dict(state_dict)
        envs.obs_rms_dict[a] = obs_rms

    episode_rewards: Dict[str, List[float]] = {a: [] for a in envs.agents}
    observations = envs.reset()

    while len(episode_rewards[envs.agents[0]]) < min_num_episodes:

        if render_mode == "rgb_array":
            yield envs.render()

        # set actions
        actions = {}
        for a, agent in agents.items():
            with torch.no_grad():
                _, action, _ = agent.act(observations[a], deterministic=True)
            actions[a] = action

        # step
        observations, _, _, infos = envs.step(actions)

        for a, info in infos.items():
            for env_idx in range(num_processes):
                if "episode" in info[env_idx]:
                    episode_rewards[a].append(info[env_idx]["episode"]["r"])

    assert all(
        [len(episode_rewards[a]) == len(episode_rewards[envs.agents[0]]) for a in envs.agents]
    ), "The number of episodes must be equal for all agents."


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--generations", type=int, nargs="+")
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")
    parser.add_argument("--disable-tracking", action="store_true", help="Disable tracking")
    parser.add_argument("--video-path", type=str, default="output.mp4")
    parser.add_argument("--low-quality", action="store_true")

    args = parser.parse_args()

    if args.generations is None:
        args.generations = [None] * 2
    elif len(args.generations) == 1:
        args.generations = args.generations * 2
    elif len(args.generations) == 2:
        pass
    else:
        raise ValueError("Invalid number of generations.")

    structures = {}
    for a, generation in zip(get_agent_names(), args.generations):
        csv_path = os.path.join(args.experiment_dir, a, "fitnesses.csv")
        id_ = get_top_robot_ids(csv_path, generation=generation)[0]
        save_path = get_robot_save_path(os.path.join(args.experiment_dir, a), id_, generation)
        print(f"Loading {save_path}")
        structures[a] = Structure.from_save_path(save_path)

    # structures = {
    #     "robot_1": Structure.from_save_path("./experiments/old/log9/robot_1"),
    #     "robot_2": Structure.from_save_path("./experiments/old/log9/robot_2"),
    # }

    coea_args = load_args(os.path.join(args.experiment_dir, "metadata"))
    env_name = coea_args.env_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_options = {"disable_tracking": True} if args.disable_tracking else None

    gen = visualize(
        structures,
        env_name,
        num_processes=1,
        device=device,
        min_num_episodes=1,
        render_mode=args.render_mode,
        render_options=render_options,
    )

    if args.render_mode == "rgb_array":
        video_path = args.video_path
        fps = 50
        frame_size = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 フォーマット用
        writer = None

    for frame in gen:
        if args.render_mode == "rgb_array":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if args.low_quality:
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            if writer is None:
                frame_size = frame.shape[1], frame.shape[0]
                writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
            writer.write(frame)
        else:
            pass

    if args.render_mode == "rgb_array" and writer is not None:
        writer.release()
