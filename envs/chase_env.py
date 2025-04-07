from copy import copy
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2  # type: ignore
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


# class SimpleChaseEnvClass(MultiAgentEvoGymBase):

#     ENV_NAME = "Chaser-v0"
#     ADDITIONAL_OBS_DIM = 7
#     PIXEL_PER_VOXEL = 30
#     WORLD_WIDTH = 400
#     DISTANCE_THRESH = 0.002 * MultiAgentEvoGymBase.VOXEL_SIZE
#     COMPLETION_REWARD = 1.0

#     def __init__(
#         self,
#         body_1: np.ndarray,
#         body_2: np.ndarray,
#         connections_1: Optional[np.ndarray] = None,
#         connections_2: Optional[np.ndarray] = None,
#         render_mode: Optional[str] = None,
#         render_options: Optional[Dict[str, Any]] = None,
#     ):

#         body_list = [body_1, body_2]
#         connections_list = [connections_1, connections_2]
#         env_file_name = "Chaser-v0.json"
#         x_positions = [188, 208]
#         # x_positions = [208, 188]
#         y_positions = [1, 1]

#         super().__init__(
#             body_list,
#             connections_list,
#             env_file_name,
#             render_mode,
#             render_options,
#             x_positions,
#             y_positions,
#         )

#         if render_options is not None and render_options["disable_tracking"]:
#             # TODO: ウィンドウサイズも設定する
#             self.default_viewer.set_pos((17.5, 4))
#         else:
#             self.default_viewer.set_pos((200, 4))
#             self.default_viewer.set_view_size((400, 20))
#             self.default_viewer.set_resolution((12000, 600))
#             # self.default_viewer.track_objects(*self.possible_agents)

#     def calc_obs(
#         self,
#         x_distances_final: Optional[Dict[str, float]] = None,
#     ) -> ObsDict:

#         robot_vel_final = [self.object_vel_at_time(self.get_time(), obj) for obj in self.agents]
#         robot_com_vel_final = [np.mean(vel, 1) for vel in robot_vel_final]

#         if x_distances_final is None:
#             robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#             x_distances_final = self.get_x_distances(robot_pos_final)

#         obs1 = np.concatenate(
#             (
#                 np.array(
#                     [
#                         robot_com_vel_final[0][0],
#                         robot_com_vel_final[0][1],
#                         robot_com_vel_final[1][0],
#                         robot_com_vel_final[1][1],
#                         x_distances_final["right"],
#                         x_distances_final["left"],
#                         1 if x_distances_final["right"] > x_distances_final["left"] else 0,
#                     ]
#                 ),
#                 self.get_relative_pos_obs(self.agents[0]),
#             )
#         )

#         obs2 = np.concatenate(
#             (
#                 np.array(
#                     [
#                         robot_com_vel_final[1][0],
#                         robot_com_vel_final[1][1],
#                         robot_com_vel_final[0][0],
#                         robot_com_vel_final[0][1],
#                         x_distances_final["right"],
#                         x_distances_final["left"],
#                         1 if x_distances_final["left"] > x_distances_final["right"] else 0,
#                     ]
#                 ),
#                 self.get_relative_pos_obs(self.agents[1]),
#             )
#         )

#         obs = {self.agents[0]: obs1, self.agents[1]: obs2}

#         return obs

#     def get_x_distances(self, robot_pos: List[np.ndarray]) -> Dict[str, float]:

#         # robot_com_pos = [np.mean(pos, 1) for pos in robot_pos]

#         warped_robot_x_bounds = [
#             {
#                 "left": self.VOXEL_SIZE * ((np.min(pos[0]) / self.VOXEL_SIZE) % 40),
#                 "right": self.VOXEL_SIZE * ((np.max(pos[0]) / self.VOXEL_SIZE) % 40),
#             }
#             for pos in robot_pos
#         ]

#         # warped_robot_com_pos = [self.VOXEL_SIZE * ((pos[0] / self.VOXEL_SIZE) % 40) for pos in robot_com_pos]

#         left_x_distance = warped_robot_x_bounds[0]["left"] - warped_robot_x_bounds[1]["right"]
#         if left_x_distance < 0:
#             if self.pre_x_distances is None:
#                 left_x_distance += 40 * self.VOXEL_SIZE
#             elif abs(left_x_distance + 40 * self.VOXEL_SIZE - self.pre_x_distances["left"]) < 5 * self.VOXEL_SIZE:
#                 left_x_distance += 40 * self.VOXEL_SIZE

#         right_x_distance = warped_robot_x_bounds[1]["left"] - warped_robot_x_bounds[0]["right"]
#         if right_x_distance < 0:
#             if self.pre_x_distances is None:
#                 right_x_distance += 40 * self.VOXEL_SIZE
#             elif abs(right_x_distance + 40 * self.VOXEL_SIZE - self.pre_x_distances["right"]) < 5 * self.VOXEL_SIZE:
#                 right_x_distance += 40 * self.VOXEL_SIZE

#         # if self.pre_x_distances is not None:

#         self.pre_x_distances = {"left": left_x_distance, "right": right_x_distance}

#         # if warped_robot_com_pos[0] < warped_robot_com_pos[1]:
#         #     right_x_distance = warped_robot_x_bounds[1]["left"] - warped_robot_x_bounds[0]["right"]
#         #     if warped_robot_x_bounds[0]["left"] > warped_robot_x_bounds[1]["right"]:
#         #         left_x_distance = warped_robot_x_bounds[0]["left"] - warped_robot_x_bounds[1]["right"]
#         #     else:
#         #         left_x_distance = (
#         #             40 * self.VOXEL_SIZE - warped_robot_x_bounds[1]["right"] + warped_robot_x_bounds[0]["left"]
#         #         )
#         # else:
#         #     if warped_robot_x_bounds[0]["right"] < warped_robot_x_bounds[1]["left"]:
#         #         right_x_distance = warped_robot_x_bounds[1]["left"] - warped_robot_x_bounds[0]["right"]
#         #     else:
#         #         right_x_distance = (
#         #             40 * self.VOXEL_SIZE - warped_robot_x_bounds[0]["right"] + warped_robot_x_bounds[1]["left"]
#         #         )
#         #     left_x_distance = warped_robot_x_bounds[0]["left"] - warped_robot_x_bounds[1]["right"]

#         # print(self.get_time(), left_x_distance, right_x_distance, abs(right_x_distance - self.pre_x_distances["right"]))

#         return {"left": left_x_distance, "right": right_x_distance}

#     def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

#         assert self.timestep is not None, "You must call reset before calling step"

#         robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#         robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]

#         is_unstable = super(MultiAgentEvoGymBase, self).step(action)

#         robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#         robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]

#         robot_vel_final = [self.object_vel_at_time(self.get_time(), obj) for obj in self.agents]
#         robot_com_vel_final = [np.mean(vel, 1) for vel in robot_vel_final]
#         print(robot_com_vel_final[0][0], robot_com_vel_final[1][0])

#         x_distances_init = self.get_x_distances(robot_pos_init)
#         x_distances_final = self.get_x_distances(robot_pos_final)

#         rewards = {a: 0.0 for a in self.agents}

#         if x_distances_init["right"] < x_distances_init["left"]:
#             rewards[self.agents[0]] += robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
#             rewards[self.agents[1]] += robot_com_pos_final[1][0] - robot_com_pos_init[1][0]
#         else:
#             rewards[self.agents[0]] -= robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
#             rewards[self.agents[1]] -= robot_com_pos_final[1][0] - robot_com_pos_init[1][0]

#         terminations = {a: True for a in self.agents}
#         if x_distances_final["right"] < self.DISTANCE_THRESH or x_distances_final["left"] < self.DISTANCE_THRESH:
#             rewards[self.agents[0]] += self.COMPLETION_REWARD
#             rewards[self.agents[1]] -= self.COMPLETION_REWARD
#         elif is_unstable:
#             print("Simulation is unstable")
#             for a in self.agents:
#                 rewards[a] -= self.COMPLETION_REWARD
#         else:
#             terminations = {a: False for a in self.agents}

#         # if x_distances_final["right"] < self.DISTANCE_THRESH or x_distances_final["left"] < self.DISTANCE_THRESH:
#         #     rewards[self.agents[0]] += self.COMPLETION_REWARD
#         #     rewards[self.agents[1]] -= self.COMPLETION_REWARD
#         # elif x_distances_init["right"] < x_distances_init["left"]:
#         #     rewards[self.agents[0]] += -(x_distances_final["right"] - x_distances_init["right"])
#         #     rewards[self.agents[1]] += x_distances_final["right"] - x_distances_init["right"]
#         #     terminations = {a: False for a in self.agents}
#         # elif x_distances_init["left"] <= x_distances_init["right"]:
#         #     rewards[self.agents[0]] += -(x_distances_final["left"] - x_distances_init["left"])
#         #     rewards[self.agents[1]] += x_distances_final["left"] - x_distances_init["left"]
#         #     terminations = {a: False for a in self.agents}
#         # elif is_unstable:
#         #     print("Simulation is unstable")
#         #     for a in self.agents:
#         #         rewards[a] -= self.COMPLETION_REWARD
#         # else:
#         #     terminations = {a: False for a in self.agents}

#         if self.timestep >= 1000:
#             rewards[self.agents[0]] -= self.COMPLETION_REWARD
#             rewards[self.agents[1]] += self.COMPLETION_REWARD
#             truncations = {a: True for a in self.agents}
#         else:
#             truncations = {a: False for a in self.agents}
#         self.timestep += 1

#         obs = self.calc_obs(x_distances_final)

#         infos: InfoDict = {a: {} for a in self.agents}
#         if all(terminations.values()) or all(truncations.values()):
#             self.agents = []

#         return obs, rewards, terminations, truncations, infos

#     def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

#         self.pre_x_distances = None
#         self.agents = copy(self.possible_agents)
#         self.timestep = 0

#         super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
#         obs = self.calc_obs()
#         infos: InfoDict = {a: {} for a in self.agents}

#         return obs, infos

#     def render(self):
#         if self._render_mode == "human":
#             super().render()
#         elif self._render_mode == "rgb_array":

#             robot_pos = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#             robot_x_bounds = [
#                 (
#                     int(np.floor(self.PIXEL_PER_VOXEL * min(pos[0]) / self.VOXEL_SIZE)),
#                     int(np.ceil(self.PIXEL_PER_VOXEL * max(pos[0]) / self.VOXEL_SIZE)),
#                 )
#                 for pos in robot_pos
#             ]

#             frame = super().render()

#             ret_frame = frame[:, :1200]
#             for bound in robot_x_bounds:
#                 left, right = bound
#                 ret_left, ret_right = left % 1200, right % 1200
#                 if ret_left < ret_right:
#                     ret_frame[:, ret_left:ret_right] = frame[:, left:right]
#                 else:
#                     left_diff = 1200 - ret_left
#                     right_diff = ret_right
#                     ret_frame[:, ret_left:] = frame[:, left : left + left_diff]
#                     ret_frame[:, :ret_right] = frame[:, right - right_diff : right]

#             return ret_frame


class SimpleChaseEnvClass(MultiAgentEvoGymBase):

    ENV_NAME = "Chaser-v0"
    ADDITIONAL_OBS_DIM = 4
    PIXEL_PER_VOXEL = 30
    WORLD_WIDTH = 400
    DISTANCE_THRESH = 0.002 * MultiAgentEvoGymBase.VOXEL_SIZE
    # COMPLETION_REWARD = 1.0
    COMPLETION_REWARD = 0.0
    MEAN_FLIP_INTERVAL = 230

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
        env_file_name = "Chaser-v0.json"
        x_positions = [188, 208]
        # x_positions = [208, 188]
        y_positions = [1, 1]

        super().__init__(
            body_list,
            connections_list,
            env_file_name,
            render_mode,
            render_options,
            x_positions,
            y_positions,
        )

        if render_options is not None and render_options["disable_tracking"]:
            # TODO: ウィンドウサイズも設定する
            self.default_viewer.set_pos((17.5, 4))
        else:
            self.default_viewer.set_pos((200, 4))
            self.default_viewer.set_view_size((400, 20))
            self.default_viewer.set_resolution((12000, 600))
            # self.default_viewer.track_objects(*self.possible_agents)

        self.direction: Optional[Literal["left", "right"]] = None
        self.pre_x_distances: Optional[Dict[str, float]] = None

    def calc_obs(
        self,
        x_distances_final: Optional[Dict[str, float]] = None,
    ) -> ObsDict:

        robot_vel_final = [self.object_vel_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_vel_final = [np.mean(vel, 1) for vel in robot_vel_final]

        if x_distances_final is None:
            robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
            x_distances_final = self.get_x_distances(robot_pos_final)

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        robot_com_vel_final[0][0],
                        robot_com_vel_final[0][1],
                        1 if self.direction == "right" else 0,
                        x_distances_final["right"] if self.direction == "right" else -x_distances_final["left"],
                        # self.count,
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
                        1 if self.direction == "left" else 0,
                        x_distances_final["left"] if self.direction == "left" else -x_distances_final["right"],
                        # self.count,
                    ]
                ),
                self.get_relative_pos_obs(self.agents[1]),
            )
        )

        obs = {self.agents[0]: obs1, self.agents[1]: obs2}

        return obs

    def get_x_distances(self, robot_pos: List[np.ndarray]) -> Dict[str, float]:

        warped_robot_x_bounds = [
            {
                "left": self.VOXEL_SIZE * ((np.min(pos[0]) / self.VOXEL_SIZE) % 40),
                "right": self.VOXEL_SIZE * ((np.max(pos[0]) / self.VOXEL_SIZE) % 40),
            }
            for pos in robot_pos
        ]

        left_x_distance = warped_robot_x_bounds[0]["left"] - warped_robot_x_bounds[1]["right"]
        if left_x_distance < 0:
            if self.pre_x_distances is None:
                left_x_distance += 40 * self.VOXEL_SIZE
            elif abs(left_x_distance + 40 * self.VOXEL_SIZE - self.pre_x_distances["left"]) < 5 * self.VOXEL_SIZE:
                left_x_distance += 40 * self.VOXEL_SIZE

        right_x_distance = warped_robot_x_bounds[1]["left"] - warped_robot_x_bounds[0]["right"]
        if right_x_distance < 0:
            if self.pre_x_distances is None:
                right_x_distance += 40 * self.VOXEL_SIZE
            elif abs(right_x_distance + 40 * self.VOXEL_SIZE - self.pre_x_distances["right"]) < 5 * self.VOXEL_SIZE:
                right_x_distance += 40 * self.VOXEL_SIZE

        self.pre_x_distances = {"left": left_x_distance, "right": right_x_distance}

        return {"left": left_x_distance, "right": right_x_distance}

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        assert self.timestep is not None, "You must call reset before calling step"

        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]

        is_unstable = super(MultiAgentEvoGymBase, self).step(action)

        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]

        x_distances_init = self.get_x_distances(robot_pos_init)
        x_distances_final = self.get_x_distances(robot_pos_final)

        rewards = {a: 0.0 for a in self.agents}

        if self.direction == "left":
            rewards[self.agents[0]] -= x_distances_final["left"] - x_distances_init["left"]
            rewards[self.agents[1]] += x_distances_final["left"] - x_distances_init["left"]
        else:
            rewards[self.agents[0]] -= x_distances_final["right"] - x_distances_init["right"]
            rewards[self.agents[1]] += x_distances_final["right"] - x_distances_init["right"]

        # if self.direction == "right":
        #     rewards[self.agents[0]] += robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
        #     rewards[self.agents[1]] += robot_com_pos_final[1][0] - robot_com_pos_init[1][0]
        # else:
        #     rewards[self.agents[0]] -= robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
        #     rewards[self.agents[1]] -= robot_com_pos_final[1][0] - robot_com_pos_init[1][0]

        terminations = {a: True for a in self.agents}

        if x_distances_final["right"] < self.DISTANCE_THRESH:
            if self.direction == "right":
                rewards[self.agents[0]] += self.COMPLETION_REWARD
                rewards[self.agents[1]] -= self.COMPLETION_REWARD
            else:
                rewards[self.agents[1]] += self.COMPLETION_REWARD
                rewards[self.agents[0]] -= self.COMPLETION_REWARD
        elif x_distances_final["left"] < self.DISTANCE_THRESH:
            if self.direction == "left":
                rewards[self.agents[0]] += self.COMPLETION_REWARD
                rewards[self.agents[1]] -= self.COMPLETION_REWARD
            else:
                rewards[self.agents[1]] += self.COMPLETION_REWARD
                rewards[self.agents[0]] -= self.COMPLETION_REWARD
        elif is_unstable:
            print("Simulation is unstable")
            for a in self.agents:
                rewards[a] -= self.COMPLETION_REWARD
        else:
            terminations = {a: False for a in self.agents}

        truncations = {a: False for a in self.agents}
        if self.timestep >= 1000:
            truncations = {a: True for a in self.agents}
        elif self.timestep == self.flip_timestep:
            self.direction = "left" if self.direction == "right" else "right"
            self.flip_timestep += np.random.poisson(self.MEAN_FLIP_INTERVAL)
            self.count += 1
        self.timestep += 1

        obs = self.calc_obs(x_distances_final)

        infos: InfoDict = {a: {} for a in self.agents}
        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

        self.pre_x_distances = None
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.count = 0
        self.direction = np.random.choice(["left", "right"])
        self.flip_timestep = np.random.poisson(self.MEAN_FLIP_INTERVAL)

        super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
        obs = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        return obs, infos

    def render(self):
        if self._render_mode == "human":
            super().render()
        elif self._render_mode == "rgb_array":

            robot_pos = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
            robot_x_bounds = [
                (
                    int(np.floor(self.PIXEL_PER_VOXEL * min(pos[0]) / self.VOXEL_SIZE)),
                    int(np.ceil(self.PIXEL_PER_VOXEL * max(pos[0]) / self.VOXEL_SIZE)),
                )
                for pos in robot_pos
            ]

            frame = super().render()

            ret_frame = frame[:, :1200]
            for bound in robot_x_bounds:
                left, right = bound
                ret_left, ret_right = left % 1200, right % 1200
                if ret_left < ret_right:
                    ret_frame[:, ret_left:ret_right] = frame[:, left:right]
                else:
                    left_diff = 1200 - ret_left
                    right_diff = ret_right
                    ret_frame[:, ret_left:] = frame[:, left : left + left_diff]
                    ret_frame[:, :ret_right] = frame[:, right - right_diff : right]

            arrow = "->" if self.direction == "right" else "<-"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(arrow, font, font_scale, font_thickness)[0]
            text_x = (ret_frame.shape[1] - text_size[0]) // 2
            text_y = text_size[1] + 10
            cv2.putText(ret_frame, arrow, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

            return ret_frame


# class SimpleTagEnvClass(MultiAgentEvoGymBase):

#     ENV_NAME = "Tag-v0"
#     DISTNACE_THRESH = 0.002 * MultiAgentEvoGymBase.VOXEL_SIZE
#     COMPLETION_REWARD = 10.0
#     ADDITIONAL_OBS_DIM = 5
#     # ADDITIONAL_OBS_DIM = 4
#     MEAN_FLIP_INTERVAL = 230

#     def __init__(
#         self,
#         body_1: np.ndarray,
#         body_2: np.ndarray,
#         connections_1: Optional[np.ndarray] = None,
#         connections_2: Optional[np.ndarray] = None,
#         render_mode: Optional[str] = None,
#         render_options: Optional[Dict[str, Any]] = None,
#     ):

#         body_list = [body_1, body_2]
#         connections_list = [connections_1, connections_2]
#         env_file_name = "Chaser-v0.json"
#         x_positions = [188, 208]
#         # x_positions = [13, 42]
#         y_positions = [1, 1]

#         super().__init__(
#             body_list,
#             connections_list,
#             env_file_name,
#             render_mode,
#             render_options,
#             x_positions,
#             y_positions,
#         )

#         self.default_viewer.track_objects(*self.possible_agents)

#     def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

#         assert self.timestep is not None, "You must call reset before calling step"

#         robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#         robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]
#         robot_x_distance_init = robot_com_pos_init[1][0] - robot_com_pos_init[0][0]

#         is_unstable = super(MultiAgentEvoGymBase, self).step(action)

#         robot_vel_final = [self.object_vel_at_time(self.get_time(), obj) for obj in self.agents]
#         robot_com_vel_final = [np.mean(vel, 1) for vel in robot_vel_final]

#         robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#         robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]
#         robot_x_distance_final = robot_com_pos_final[1][0] - robot_com_pos_final[0][0]

#         rewards = {a: 0.0 for a in self.agents}

#         if self.tagger == 0:
#             rewards[self.agents[0]] -= robot_x_distance_final - robot_x_distance_init
#             rewards[self.agents[1]] += robot_x_distance_final - robot_x_distance_init
#         else:
#             rewards[self.agents[0]] += robot_x_distance_final - robot_x_distance_init
#             rewards[self.agents[1]] -= robot_x_distance_final - robot_x_distance_init

#         gap = min(robot_pos_final[1][0]) - max(robot_pos_final[0][0])

#         # if gap < self.DISTNACE_THRESH and not self.flag:
#         #     self.flag = True
#         #     # self.count += 1
#         #     self.tagger = 0
#         #     print("flipped")

#         # if self.tagger == 0:
#         #     rewards[self.agents[0]] += robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
#         #     rewards[self.agents[1]] += robot_com_pos_final[1][0] - robot_com_pos_init[1][0]
#         # else:
#         #     rewards[self.agents[0]] -= robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
#         #     rewards[self.agents[1]] -= robot_com_pos_final[1][0] - robot_com_pos_init[1][0]

#         if gap < self.DISTNACE_THRESH:
#             if self.tagger == 0:
#                 rewards[self.agents[0]] += self.COMPLETION_REWARD
#                 rewards[self.agents[1]] -= self.COMPLETION_REWARD
#             else:
#                 rewards[self.agents[0]] -= self.COMPLETION_REWARD
#                 rewards[self.agents[1]] += self.COMPLETION_REWARD
#             terminations = {a: True for a in self.agents}
#         elif is_unstable:
#             print("Simulation is unstable")
#             for a in self.agents:
#                 rewards[a] -= self.COMPLETION_REWARD
#             terminations = {a: True for a in self.agents}
#         else:
#             terminations = {a: False for a in self.agents}

#         if self.timestep >= 1000:
#             truncations = {a: True for a in self.agents}
#         elif self.timestep == self.flip_timestep:
#             if self.tagger == 0:
#                 self.tagger = 1
#             else:
#                 self.tagger = 0
#             self.flip_timestep += np.random.poisson(self.MEAN_FLIP_INTERVAL)
#             self.count += 1
#             truncations = {a: False for a in self.agents}
#         else:
#             truncations = {a: False for a in self.agents}
#         self.timestep += 1

#         obs = self.calc_obs(robot_com_pos_final)
#         infos: InfoDict = {a: {} for a in self.agents}

#         if all(terminations.values()) or all(truncations.values()):
#             self.agents = []

#         return obs, rewards, terminations, truncations, infos

#     def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

#         self.agents = copy(self.possible_agents)
#         self.timestep = 0
#         self.count = 0
#         self.flag = False
#         self.tagger = np.random.choice([0, 1])
#         self.tagger = 0
#         self.flip_timestep = np.random.poisson(self.MEAN_FLIP_INTERVAL)

#         super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
#         obs = self.calc_obs()
#         infos: InfoDict = {a: {} for a in self.agents}

#         return obs, infos

#     def calc_obs(
#         self,
#         robot_com_pos_final: Optional[List[np.ndarray]] = None,
#     ) -> ObsDict:

#         robot_vel_final = [self.object_vel_at_time(self.get_time(), obj) for obj in self.agents]
#         robot_com_vel_final = [np.mean(vel, 1) for vel in robot_vel_final]

#         if robot_com_pos_final is None:
#             robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#             robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]

#         robot_x_distance = robot_com_pos_final[1][0] - robot_com_pos_final[0][0]

#         obs1 = np.concatenate(
#             (
#                 np.array(
#                     [
#                         robot_com_vel_final[0][0],
#                         robot_com_vel_final[0][1],
#                         1 if self.tagger == 0 else 0,
#                         robot_x_distance if self.tagger == 0 else -robot_x_distance,
#                         self.count,
#                     ]
#                 ),
#                 self.get_relative_pos_obs(self.agents[0]),
#             )
#         )

#         obs2 = np.concatenate(
#             (
#                 np.array(
#                     [
#                         robot_com_vel_final[1][0],
#                         robot_com_vel_final[1][1],
#                         1 if self.tagger == 1 else 0,
#                         robot_x_distance if self.tagger == 1 else -robot_x_distance,
#                         self.count,
#                     ]
#                 ),
#                 self.get_relative_pos_obs(self.agents[1]),
#             )
#         )

#         obs = {self.agents[0]: obs1, self.agents[1]: obs2}

#         return obs
