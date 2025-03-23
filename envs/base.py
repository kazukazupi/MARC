import functools
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from evogym import EvoWorld  # type: ignore
from evogym.envs import EvoGymBase  # type: ignore
from gymnasium import spaces  # type: ignore
from pettingzoo import ParallelEnv  # type: ignore

from envs.typehints import ActionDict, BoolDict, InfoDict, ObsDict, RewardDict
from utils import get_agent_names


class MultiAgentEvoGymBase(EvoGymBase, ParallelEnv):

    VOXEL_SIZE = 0.1
    ADDITIONAL_OBS_DIM: int
    ENV_NAME: str

    def __init__(
        self,
        body_list: List[np.ndarray],
        connections_list: List[Optional[np.ndarray]],
        env_file_name: str,
        render_mode: Optional[str],
        render_options: Optional[dict],
        x_positions: List[int],
        y_positions: List[int],
    ):

        self.possible_agents = get_agent_names()
        self.agents: List[str] = []
        self.timestep: Optional[int] = None

        # make world
        self.world = EvoWorld.from_json(os.path.join("world_data", env_file_name))

        for agent, body, connections, x, y in zip(
            self.possible_agents, body_list, connections_list, x_positions, y_positions
        ):

            self.world.add_from_array(agent, body, x, y, connections=connections)

        EvoGymBase.__init__(self, self.world, render_mode, render_options)

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:
        raise NotImplementedError

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        num_robot_points = self.object_pos_at_time(self.get_time(), agent).size

        return spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(self.ADDITIONAL_OBS_DIM + num_robot_points,),
            dtype=float,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):

        num_actuators = self.get_actuator_indices(agent).size

        return spaces.Box(
            low=0.6,
            high=1.6,
            shape=(num_actuators,),
            dtype=float,
        )

    def pos_at_time(self, time):
        return super().pos_at_time(time) * self.VOXEL_SIZE

    def vel_at_time(self, time):
        return super().vel_at_time(time) * self.VOXEL_SIZE

    def object_pos_at_time(self, time, object_name):
        return super().object_pos_at_time(time, object_name) * self.VOXEL_SIZE

    def object_vel_at_time(self, time, object_name):
        return super().object_vel_at_time(time, object_name) * self.VOXEL_SIZE

    def get_pos_com_obs(self, object_name):
        return super().get_pos_com_obs(object_name) * self.VOXEL_SIZE

    def get_vel_com_obs(self, object_name):
        temp = super().get_vel_com_obs(object_name) * self.VOXEL_SIZE
        # print(f'child says super vel obs: {super().get_vel_com_obs(object_name)}\n')
        # print(f'vel obs: {temp}\n\n')
        return temp

    def get_relative_pos_obs(self, object_name):
        return super().get_relative_pos_obs(object_name) * self.VOXEL_SIZE

    def get_floor_obs(self, object_name, terrain_list, sight_dist, sight_range=5):
        return super().get_floor_obs(object_name, terrain_list, sight_dist, sight_range) * self.VOXEL_SIZE

    def get_ceil_obs(self, object_name, terrain_list, sight_dist, sight_range=5):
        return super().get_ceil_obs(object_name, terrain_list, sight_dist, sight_range) * self.VOXEL_SIZE
