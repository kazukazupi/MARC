from typing import Any, Dict, List, Optional, Union

import cv2  # type: ignore
import numpy as np
import torch

from alg.coea.structure import DummyRobotStructure, Structure
from alg.controller import Controller
from alg.ppo.controller import AgentController
from alg.ppo.env_wrappers import MultiAgentVecNormalize, make_multi_agent_vec_envs
from utils import AGENT_1, AGENT_2, AgentID


def evaluate(
    structures: Dict[AgentID, Union[Structure, DummyRobotStructure]],
    env_name: str,
    num_processes: int,
    device: torch.device,
    min_num_episodes: int = 1,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    render_options: Optional[Dict[str, Any]] = None,
    movie_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    制御器を評価する関数

    Parameters
    ----------
    structures : Dict[AgentID, Union[Structure, DummyRobotStructure]]
        各エージェントの構造
    env_name : str
        環境名
    num_processes : int
        並列環境数
    device : torch.device
        制御器のロード時に使用するデバイス
    min_num_episodes : int
        最小エピソード数
    seed : Optional[int]
        シード値
    render_mode : Optional[str]
        レンダリングモード
    render_options : Optional[Dict[str, Any]]
        レンダリングオプション
    movie_path : Optional[str]
        動画の保存先パス

    Returns
    -------
    Dict[str, float]
        各エージェントの評価スコア
    """

    # Create environment (numpy-based)
    envs: MultiAgentVecNormalize = make_multi_agent_vec_envs(
        env_name,
        num_processes,
        None,
        device,
        training=False,
        seed=seed,
        body_1=structures[AGENT_1].body,
        body_2=structures[AGENT_2].body,
        connections_1=structures[AGENT_1].connections,
        connections_2=structures[AGENT_2].connections,
        render_mode=render_mode,
        render_options=render_options,
        use_pytorch_wrapper=False,  # numpy-based environment
    )

    # Load controllers using Controller interface
    controllers: Dict[AgentID, Controller] = {}
    for a, structure in structures.items():
        if isinstance(structure, DummyRobotStructure):
            continue
        controller_path = structure.get_latest_controller_path()
        # Load PPO agent using AgentController wrapper
        controller, obs_rms = AgentController.from_file(controller_path, device)
        controllers[a] = controller
        envs.obs_rms_dict[a] = obs_rms

    # Initialize episode rewards
    episode_rewards: Dict[str, List[float]] = {a: [] for a in envs.agents}
    observations = envs.reset()

    if render_mode == "rgb_array":
        fps = 50
        frame_size = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        writer = None
        assert movie_path is not None, "movie_path must be provided when render_mode is 'rgb_array'"

    while len(episode_rewards[envs.agents[0]]) < min_num_episodes:

        # render
        if render_mode == "rgb_array":
            frame = envs.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if writer is None:
                frame_size = frame.shape[1], frame.shape[0]
                assert movie_path is not None
                writer = cv2.VideoWriter(movie_path, fourcc, fps, frame_size)
            writer.write(frame)

        # set actions using Controller interface (numpy-based)
        actions = {}
        for a, controller in controllers.items():
            action = controller.act(observations[a], deterministic=True)
            actions[a] = action

        # step (numpy-based)
        observations, _, _, infos = envs.step(actions)

        # reward
        for a, info in infos.items():
            for env_idx in range(num_processes):
                if "episode" in info[env_idx]:
                    episode_rewards[a].append(info[env_idx]["episode"]["r"])

    assert all(
        [len(episode_rewards[a]) == len(episode_rewards[envs.agents[0]]) for a in envs.agents]
    ), "The number of episodes must be equal for all agents."

    if render_mode == "rgb_array" and writer is not None:
        writer.release()

    if "fitness" in infos[AGENT_1][0]:
        return {a: infos[a][0]["fitness"] for a in envs.agents}

    return {a: float(np.mean(episode_rewards[a])) for a in envs.agents}
