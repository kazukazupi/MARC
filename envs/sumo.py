from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.base import MultiAgentEvoGymBase
from envs.typehints import ActionDict, BoolDict, InfoDict, ObsDict, RewardDict


class SumoEnvClass(MultiAgentEvoGymBase):

    ENV_NAME = "Sumo-v0"
    ADDITIONAL_OBS_DIM = 6
    ROBOT1_INIT_POS = (8, 3)
    ROBOT2_INIT_POS = (22, 3)

    VIEWER_DEFAULT_POS = (17.5, 4)

    COLLIDE_THRESH = 0.1 * MultiAgentEvoGymBase.VOXEL_SIZE
    HEIGHT_THRESH = 1.02 * MultiAgentEvoGymBase.VOXEL_SIZE
    COMPLETION_REWARD = 1.0

    def __init__(
        self,
        body_1: np.ndarray,
        body_2: np.ndarray,
        connections_1: Optional[np.ndarray] = None,
        connections_2: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        self.collide = False

        super().__init__(
            body_1=body_1,
            body_2=body_2,
            connections_1=connections_1,
            connections_2=connections_2,
            render_mode=render_mode,
            render_options=render_options,
        )

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        assert self.timestep is not None, "You must call reset before calling step"

        # collect pre step information
        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]

        # When this is True, the simulation has reached an unstable state from which it cannot recover
        is_unstable = super(MultiAgentEvoGymBase, self).step(action)

        # collect post step information
        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]

        # calculate positions and velocities of center of mass
        robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]
        robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]

        # calculate minimum height for each robot
        min_heights = [np.min(pos[1]) for pos in robot_pos_final]

        # calculate reward
        rewards = {a: 0.0 for a in self.agents}
        rewards[self.agents[0]] += robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
        rewards[self.agents[1]] += -(robot_com_pos_final[1][0] - robot_com_pos_init[1][0])

        # judge termination
        terminations = {a: True for a in self.agents}

        # detect collision
        if not self.collide:
            distance = np.min(robot_pos_final[1][0]) - np.max(robot_pos_final[0][0])
            if distance < self.COLLIDE_THRESH:
                self.collide = True

        if min_heights[0] < self.HEIGHT_THRESH and min_heights[1] < self.HEIGHT_THRESH:
            for a in self.agents:
                rewards[a] -= self.COMPLETION_REWARD
        elif min_heights[0] < self.HEIGHT_THRESH:
            rewards[self.agents[0]] -= self.COMPLETION_REWARD
            if self.collide:
                rewards[self.agents[1]] += self.COMPLETION_REWARD
        elif min_heights[1] < self.HEIGHT_THRESH:
            if self.collide:
                rewards[self.agents[0]] += self.COMPLETION_REWARD
            rewards[self.agents[1]] -= self.COMPLETION_REWARD
        elif is_unstable:
            print("SIMULATION UNSTABLE... TERMINATING")
            for a in self.agents:
                rewards[a] -= self.COMPLETION_REWARD
        else:
            terminations = {a: False for a in self.agents}

        # judge truncation
        if self.timestep >= 1000:
            rewards = {a: 0.0 for a in self.agents}
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}
        self.timestep += 1

        # return
        obs = self.calc_obs(robot_pos_final, robot_com_pos_final)
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

        self.collide = False
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
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
