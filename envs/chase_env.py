from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.base import MultiAgentEvoGymBase
from envs.typehints import ActionDict, BoolDict, InfoDict, ObsDict, RewardDict


class HorizontalChaseEnvClass(MultiAgentEvoGymBase):

    def __init__(
        self,
        body_1: np.ndarray,
        body_2: np.ndarray,
        connections_1: Optional[np.ndarray] = None,
        connections_2: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        body_list = [body_1, body_2]
        connections_list = [connections_1, connections_2]
        env_file_name = "h_chase_env.json"
        x_positions = [15 - body_1.shape[1] // 2, 15 - body_2.shape[1] // 2]
        y_positions = [1, 9]

        super().__init__(
            body_list,
            connections_list,
            env_file_name,
            render_mode,
            render_options,
            x_positions,
            y_positions,
        )

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]

        assert self.timestep is not None, "You must call reset before calling step"

        is_unstable = super(MultiAgentEvoGymBase, self).step(action)

        if is_unstable:
            print("Simulation is unstable")

        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]

        robots_abs_distance_x_init = np.abs(robot_com_pos_init[1][0] - robot_com_pos_init[0][0])
        robots_abs_distance_x_final = np.abs(robot_com_pos_final[1][0] - robot_com_pos_final[0][0])

        rewards = {a: 0.0 for a in self.agents}
        rewards[self.agents[0]] -= (
            robots_abs_distance_x_final - robots_abs_distance_x_init
        )  # The closer together, the better
        rewards[self.agents[1]] += (
            robots_abs_distance_x_final - robots_abs_distance_x_init
        )  # The farther apart, the better

        terminations = {a: False for a in self.agents}

        if self.timestep >= 1000:
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}
        self.timestep += 1

        obs = self.calc_obs(robot_pos_final, robot_com_pos_final)
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

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
