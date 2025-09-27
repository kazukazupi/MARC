import functools
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from evogym import EvoWorld  # type: ignore
from evogym.envs import EvoGymBase  # type: ignore
from gymnasium import spaces  # type: ignore
from pettingzoo import ParallelEnv  # type: ignore

from envs.typehints import ActionDict, BoolDict, InfoDict, ObsDict, RewardDict
from utils import AGENT_IDS, AgentID


class MultiAgentEvoGymBase(EvoGymBase, ParallelEnv):

    VOXEL_SIZE = 0.1

    ENV_NAME: str
    ADDITIONAL_OBS_DIM: int
    ROBOT1_INIT_POS: Tuple[int, int]
    ROBOT2_INIT_POS: Tuple[int, int]

    VIEWER_DEFAULT_POS: Tuple[float, float]

    def __init__(
        self,
        body_1: np.ndarray,
        body_2: np.ndarray,
        connections_1: Optional[np.ndarray] = None,
        connections_2: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        self.possible_agents: List[AgentID] = AGENT_IDS
        self.agents: List[AgentID] = []
        self.timestep: Optional[int] = None

        # make world
        self.world = EvoWorld.from_json(os.path.join("envs", "world_data", self.ENV_NAME + ".json"))

        # add robot1 to world
        self.world.add_from_array(
            self.possible_agents[0],
            body_1,
            self.ROBOT1_INIT_POS[0],
            self.ROBOT1_INIT_POS[1],
            connections=connections_1,
        )

        # add robot2 to world
        self.world.add_from_array(
            self.possible_agents[1],
            body_2,
            self.ROBOT2_INIT_POS[0],
            self.ROBOT2_INIT_POS[1],
            connections=connections_2,
        )

        EvoGymBase.__init__(self, self.world, render_mode, render_options)

        # viewer setup
        if render_options is not None and render_options["disable_tracking"]:
            self.default_viewer.set_pos(self.VIEWER_DEFAULT_POS)
        else:
            self.default_viewer.track_objects(*self.possible_agents)

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
