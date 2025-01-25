import functools
import os
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from evogym import EvoWorld
from evogym.envs import EvoGymBase
from gymnasium import spaces
from pettingzoo import ParallelEnv

ObsType = np.ndarray
ActionType = np.ndarray
AgentID = str

ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[AgentID, ActionType]
RewardDict = Dict[AgentID, float]
BoolDict = Dict[AgentID, bool]
InfoDict = Dict[AgentID, Dict[str, Any]]


class SimpleSumoEnvClass(EvoGymBase, ParallelEnv):

    metadata = {
        "name": "Sumo-v0",
    }

    def __init__(
        self,
        body_1: np.ndarray,
        body_2: np.ndarray,
        connections_1: Optional[np.ndarray] = None,
        connections_2: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        self.possible_agents = ["robot_1", "robot_2"]

        # make world
        self.world = EvoWorld.from_json(os.path.join("world_data", "simple_sumo_env.json"))
        self.world.add_from_array(self.possible_agents[0], body_1, 15 - body_1.shape[1], 1, connections=connections_1)
        self.world.add_from_array(self.possible_agents[1], body_2, 16, 1, connections=connections_2)

        # init sim
        EvoGymBase.__init__(self, self.world, render_mode, render_options)

        # set viewer
        self.default_viewer.track_objects(*self.possible_agents)

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        # collect pre step information
        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]

        # When this is True, the simulation has reached an unstable state from which it cannot recover
        done = super().step(action)

        # collect post step information
        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]

        # calculate positions and velocities of center of mass
        robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]
        robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]

        # calculate reward
        rewards = {a: 0.0 for a in self.agents}
        rewards[self.agents[0]] += robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
        rewards[self.agents[1]] += -(robot_com_pos_final[1][0] - robot_com_pos_init[1][0])

        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
        if robot_com_pos_final[0][0] > 28:
            done = True
            rewards[self.agents[0]] -= 1.0
            rewards[self.agents[1]] += 1.0
        if robot_com_pos_final[1][0] < 2:
            done = True
            rewards[self.agents[0]] += 1.0
            rewards[self.agents[1]] -= 1.0

        obs = self.calc_obs(robot_pos_final, robot_com_pos_final)
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos: InfoDict = {a: {} for a in self.agents}

        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

        self.agents = copy(self.possible_agents)

        super().reset(seed=seed, options=options)
        obs = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        return obs, infos

    def calc_obs(
        self,
        robot_pos_final: Optional[List[np.ndarray]] = None,
        robot_com_pos_final: Optional[List[np.ndarray]] = None,
    ) -> ObsDict:

        # collect post step information
        if robot_pos_final is None:
            robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]

        robot_vel_final = [self.object_vel_at_time(self.get_time(), obj) for obj in self.agents]

        # calculate positions and velocities of center of mass
        if robot_com_pos_final is None:
            robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]
        robot_com_vel_final = [np.mean(vel, 1) for vel in robot_vel_final]

        # calulate observations
        robots_distance_x = robot_com_pos_final[1][0] - robot_com_pos_final[0][0]
        robots_distance_y = robot_com_pos_final[1][1] - robot_com_pos_final[0][1]

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        robot_com_vel_final[0][0],
                        robot_com_vel_final[0][1],
                        robot_com_vel_final[1][0],
                        robot_com_vel_final[1][1],
                        robots_distance_x,
                        robots_distance_y,
                    ]
                ),
                self.get_relative_pos_obs(self.agents[0]),
            )
        )

        obs2 = np.concatenate(
            (
                np.array(
                    [
                        robot_com_vel_final[1][0],
                        robot_com_vel_final[1][1],
                        robot_com_vel_final[0][0],
                        robot_com_vel_final[0][1],
                        robots_distance_x,
                        robots_distance_y,
                    ]
                ),
                self.get_relative_pos_obs(self.agents[1]),
            )
        )

        obs = {self.agents[0]: obs1, self.agents[1]: obs2}

        return obs

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        num_robot_points = self.object_pos_at_time(self.get_time(), agent).size

        return spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(6 + num_robot_points,),
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
