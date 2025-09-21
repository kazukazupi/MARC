from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.base import MultiAgentEvoGymBase
from envs.typehints import ActionDict, BoolDict, InfoDict, ObsDict, RewardDict


class ChimneyClashEnvClass(MultiAgentEvoGymBase):

    ENV_NAME = "ChimneyClash"
    ADDITIONAL_OBS_DIM = 4
    ENV_FILE_NAME = "ChimneyClash.json"
    ROBOT1_INIT_POS = (7, 1)
    ROBOT2_INIT_POS = (9, 7)

    VIEWER_DEFAULT_POS = (17.5, 4)

    Y_THRESH = 6 * MultiAgentEvoGymBase.VOXEL_SIZE
    Y_THRESH_REWARD = 8 * MultiAgentEvoGymBase.VOXEL_SIZE

    def __init__(
        self,
        body_1: np.ndarray,
        body_2: np.ndarray,
        connections_1: Optional[np.ndarray] = None,
        connections_2: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(
            body_1=body_1,
            body_2=body_2,
            connections_1=connections_1,
            connections_2=connections_2,
            render_mode=render_mode,
            render_options=render_options,
        )

        self.robot2_max_com_y: Optional[float] = None
        self.robot2_min_com_y: Optional[float] = None

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        assert self.timestep is not None, "You must call reset before calling step"

        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_init = [np.mean(pos, axis=1) for pos in robot_pos_init]
        robot2_filtered_pos_init = np.array([pos for pos in robot_pos_init[1].T if pos[1] > self.Y_THRESH_REWARD]).T
        robot2_min_pos_init = np.min(robot2_filtered_pos_init, axis=1)
        robot2_max_pos_init = np.max(robot2_filtered_pos_init, axis=1)

        is_unstable = super(MultiAgentEvoGymBase, self).step(action)

        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_final = [np.mean(pos, axis=1) for pos in robot_pos_final]
        robot2_filtered_pos_final = np.array([pos for pos in robot_pos_final[1].T if pos[1] > self.Y_THRESH_REWARD]).T
        robot2_min_pos_final = np.min(robot2_filtered_pos_final, axis=1)
        robot2_max_pos_final = np.max(robot2_filtered_pos_final, axis=1)

        if self.robot2_max_com_y < robot_com_pos_final[1][1]:
            self.robot2_max_com_y = robot_com_pos_final[1][1]
        elif self.robot2_min_com_y > robot_com_pos_final[1][1]:
            self.robot2_min_com_y = robot_com_pos_final[1][1]

        span_final = robot2_max_pos_final[0] - robot2_min_pos_final[0]
        span_init = robot2_max_pos_init[0] - robot2_min_pos_init[0]

        rewards = {a: 0.0 for a in self.agents}

        # Reward based on the elevation of robot 2
        rewards[self.agents[0]] += robot_com_pos_final[1][1] - robot_com_pos_init[1][1]
        rewards[self.agents[1]] -= robot_com_pos_final[1][1] - robot_com_pos_init[1][1]

        # Exploration reward
        rewards[self.agents[0]] += (
            abs(robot_com_pos_init[0][1] - robot_com_pos_init[1][1])
            - abs(robot_com_pos_final[0][1] - robot_com_pos_final[1][1])
        ) * 0.5
        rewards[self.agents[1]] += (span_final - span_init) * 0.5

        terminations = {a: False for a in self.agents}
        if robot_com_pos_final[1][1] < self.Y_THRESH:
            rewards[self.agents[1]] -= 1.0
            terminations = {a: True for a in self.agents}
        elif is_unstable:
            terminations = {a: True for a in self.agents}

        if self.timestep >= 500:
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}
        self.timestep += 1

        obs = self.calc_obs(robot_com_pos_final)
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            assert self.robot2_max_com_y is not None
            assert self.robot2_min_com_y is not None
            infos[self.agents[0]]["fitness"] = self.robot2_max_com_y - self.robot2_min_com_y
            infos[self.agents[1]]["fitness"] = -(self.robot2_max_com_y - self.robot2_min_com_y)
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def calc_obs(self, robot_com_pos_final: Optional[List[np.ndarray]] = None) -> ObsDict:

        if robot_com_pos_final is None:
            robot_com_pos_final = [self.get_pos_com_obs(a) for a in self.agents]

        x_distance = robot_com_pos_final[0][0] - robot_com_pos_final[1][0]
        y_distance = robot_com_pos_final[0][1] - robot_com_pos_final[1][1]

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        x_distance,
                        y_distance,
                    ]
                ),
                self.get_vel_com_obs(self.agents[0]),
                self.get_relative_pos_obs(self.agents[0]),
            )
        )

        obs2 = np.concatenate(
            (
                np.array(
                    [
                        x_distance,
                        y_distance,
                    ]
                ),
                self.get_vel_com_obs(self.agents[1]),
                self.get_relative_pos_obs(self.agents[1]),
            )
        )

        observations = {
            self.agents[0]: obs1,
            self.agents[1]: obs2,
        }

        return observations

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
        obs = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        self.robot2_min_com_y = self.get_pos_com_obs(self.agents[1])[1]
        self.robot2_max_com_y = self.get_pos_com_obs(self.agents[1])[1]

        return obs, infos
