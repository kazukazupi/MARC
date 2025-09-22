from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.base import MultiAgentEvoGymBase
from envs.typehints import ActionDict, BoolDict, InfoDict, ObsDict, RewardDict


class PackageBase(MultiAgentEvoGymBase):

    ADDITIONAL_OBS_DIM = 6

    def calc_obs(
        self,
        robot_com_pos_final: Optional[List[np.ndarray]] = None,
        package_com_pos_final: Optional[np.ndarray] = None,
    ) -> ObsDict:

        if robot_com_pos_final is None:
            robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
            robot_com_pos_final = [np.mean(pos, axis=1) for pos in robot_pos_final]

        if package_com_pos_final is None:
            package_pos_final = self.object_pos_at_time(self.get_time(), "package")
            package_com_pos_final = np.mean(package_pos_final, axis=1)

        robot_vel_final = [self.object_vel_at_time(self.get_time(), obj) for obj in self.agents]
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")
        robot_com_vel_final = [np.mean(vel, axis=1) for vel in robot_vel_final]
        package_com_vel_final = np.mean(package_vel_final, axis=1)

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        robot_com_vel_final[0][0],
                        robot_com_vel_final[0][1],
                        package_com_pos_final[0] - robot_com_pos_final[0][0],
                        package_com_pos_final[1] - robot_com_pos_final[0][1],
                        package_com_vel_final[0],
                        package_com_vel_final[1],
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
                        package_com_pos_final[0] - robot_com_pos_final[1][0],
                        package_com_pos_final[1] - robot_com_pos_final[1][1],
                        package_com_vel_final[0],
                        package_com_vel_final[1],
                    ]
                ),
                self.get_relative_pos_obs(self.agents[1]),
            )
        )

        observations = {self.agents[0]: obs1, self.agents[1]: obs2}

        return observations

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
        obs = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        return obs, infos


class ObjectPushEnvClass(PackageBase):

    ENV_NAME = "BoxPush-v0"
    ENV_FILE_NAME = "push_env.json"
    ROBOT1_INIT_POS = (8, 3)
    ROBOT2_INIT_POS = (22, 3)

    VIEWER_DEFAULT_POS = (17.5, 4)

    HEIGHT_THRESH = 1.02 * MultiAgentEvoGymBase.VOXEL_SIZE
    COMPLETION_REWARD = 1.0

    def get_rewards(
        self,
        package_com_pos_init: np.ndarray,
        package_com_pos_final: np.ndarray,
        robot_com_pos_init: List[np.ndarray],
        robot_com_pos_final: List[np.ndarray],
    ) -> RewardDict:

        rewards = {a: 0.0 for a in self.agents}

        # positive reward for moving forward
        rewards[self.agents[0]] += (package_com_pos_final[0] - package_com_pos_init[0]) * 0.75
        rewards[self.agents[1]] -= rewards[self.agents[0]]
        rewards[self.agents[0]] += (robot_com_pos_final[0][0] - robot_com_pos_init[0][0]) * 0.5
        rewards[self.agents[1]] -= (robot_com_pos_final[1][0] - robot_com_pos_init[1][0]) * 0.5

        # negative reward for robot/package separating
        rewards[self.agents[0]] += abs(robot_com_pos_init[0][0] - package_com_pos_init[0]) - abs(
            robot_com_pos_final[0][0] - package_com_pos_final[0]
        )
        rewards[self.agents[1]] += abs(robot_com_pos_init[1][0] - package_com_pos_init[0]) - abs(
            robot_com_pos_final[1][0] - package_com_pos_final[0]
        )

        return rewards

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        assert self.timestep is not None, "You must call reset before calling step"

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        package_com_pos_init = np.mean(package_pos_init, axis=1)
        robot_com_pos_init = [np.mean(pos, axis=1) for pos in robot_pos_init]

        # step
        is_unstable = super(MultiAgentEvoGymBase, self).step(action)

        # collect post step information
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        package_com_pos_final = np.mean(package_pos_final, axis=1)
        robot_com_pos_final = [np.mean(pos, axis=1) for pos in robot_pos_final]

        # calculate minimum height for each robot
        min_heights = [np.min(pos[1]) for pos in robot_pos_final]

        # calculate reward
        rewards = self.get_rewards(
            package_com_pos_init,
            package_com_pos_final,
            robot_com_pos_init,
            robot_com_pos_final,
        )

        # judge termination
        terminations = {a: True for a in self.agents}

        if min_heights[0] < self.HEIGHT_THRESH and min_heights[1] < self.HEIGHT_THRESH:
            for a in self.agents:
                rewards[a] -= self.COMPLETION_REWARD
        elif min_heights[0] < self.HEIGHT_THRESH:
            rewards[self.agents[0]] -= self.COMPLETION_REWARD
            rewards[self.agents[1]] += self.COMPLETION_REWARD
        elif min_heights[1] < self.HEIGHT_THRESH:
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
        observations = self.calc_obs(robot_com_pos_final, package_com_pos_final)
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos


class AboveObjectPushEnvClass(PackageBase):

    ENV_NAME = "AboveBoxPush-v0"
    ENV_FILE_NAME = "MultiPusher-v2.json"
    ROBOT1_INIT_POS = (7, 1)
    ROBOT2_INIT_POS = (19, 1)
    X_THRESH_1 = 14 * MultiAgentEvoGymBase.VOXEL_SIZE
    X_THRESH_2 = 17 * MultiAgentEvoGymBase.VOXEL_SIZE
    COMPLETION_REWARD = 1.0

    VIEWER_DEFAULT_POS = (17.5, 4)

    def get_rewards(
        self,
        package_com_pos_init: np.ndarray,
        package_com_pos_final: np.ndarray,
        robot_com_pos_init: List[np.ndarray],
        robot_com_pos_final: List[np.ndarray],
    ) -> RewardDict:

        rewards = {a: 0.0 for a in self.agents}

        # positive reward for package moving forward
        rewards[self.agents[0]] += package_com_pos_final[0] - package_com_pos_init[0]
        rewards[self.agents[1]] -= package_com_pos_final[0] - package_com_pos_init[0]

        # positive reward for robot approaching package
        rewards[self.agents[0]] += abs(robot_com_pos_init[0][0] - package_com_pos_init[0]) - abs(
            robot_com_pos_final[0][0] - package_com_pos_final[0]
        )
        rewards[self.agents[0]] += abs(robot_com_pos_init[0][1] - package_com_pos_init[1]) - abs(
            robot_com_pos_final[0][1] - package_com_pos_final[1]
        )
        rewards[self.agents[1]] += abs(robot_com_pos_init[1][0] - package_com_pos_init[0]) - abs(
            robot_com_pos_final[1][0] - package_com_pos_final[0]
        )
        rewards[self.agents[1]] += abs(robot_com_pos_init[1][1] - package_com_pos_init[1]) - abs(
            robot_com_pos_final[1][1] - package_com_pos_final[1]
        )

        # # positive reward for robot approaching package
        # rewards[self.agents[0]] += (robot_com_pos_final[0][0] - robot_com_pos_init[0][0]) * 0.1
        # rewards[self.agents[0]] += (robot_com_pos_final[0][1] - robot_com_pos_init[0][1]) * 0.1
        # rewards[self.agents[1]] -= (robot_com_pos_final[1][0] - robot_com_pos_init[1][0]) * 0.1
        # rewards[self.agents[1]] += (robot_com_pos_final[1][1] - robot_com_pos_init[1][1]) * 0.1

        return rewards

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        assert self.timestep is not None, "You must call reset before calling step"

        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        package_com_pos_init = np.mean(package_pos_init, axis=1)
        robot_com_pos_init = [np.mean(pos, axis=1) for pos in robot_pos_init]

        is_unstable = super(MultiAgentEvoGymBase, self).step(action)

        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        package_com_pos_final = np.mean(package_pos_final, axis=1)
        robot_com_pos_final = [np.mean(pos, axis=1) for pos in robot_pos_final]

        rewards = self.get_rewards(
            package_com_pos_init, package_com_pos_final, robot_com_pos_init, robot_com_pos_final
        )

        terminations = {a: True for a in self.agents}
        if np.min(package_pos_final[0]) > self.X_THRESH_2:
            rewards[self.agents[0]] += self.COMPLETION_REWARD
        elif np.max(package_pos_final[0]) < self.X_THRESH_1:
            rewards[self.agents[1]] += self.COMPLETION_REWARD
        elif is_unstable:
            print("SIMULATION UNSTABLE... TERMINATING")
            for a in self.agents:
                rewards[a] -= self.COMPLETION_REWARD
        else:
            terminations = {a: False for a in self.agents}

        if self.timestep >= 1000:
            rewards = {a: 0.0 for a in self.agents}
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}
        self.timestep += 1

        observations = self.calc_obs(robot_com_pos_final, package_com_pos_final)
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos


# class ObjectPullEnvClass(PackageBase):

#     ENV_NAME = "MultiPusher-v1"
#     LEFT_THRESH = 10 * MultiAgentEvoGymBase.VOXEL_SIZE
#     RIGHT_THRESH = 41 * MultiAgentEvoGymBase.VOXEL_SIZE
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
#         env_file_name = "MultiPusher-v1.json"
#         x_positions = [24 - body_1.shape[1], 27]
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
#             self.default_viewer.track_objects(*self.possible_agents)

#     def get_rewards(
#         self,
#         package_com_pos_init: np.ndarray,
#         package_com_pos_final: np.ndarray,
#         robot_com_pos_init: List[np.ndarray],
#         robot_com_pos_final: List[np.ndarray],
#     ) -> RewardDict:

#         rewards = {a: 0.0 for a in self.agents}

#         # positive reward for moving backward
#         rewards[self.agents[1]] += (package_com_pos_final[0] - package_com_pos_init[0]) * 0.75
#         rewards[self.agents[0]] -= rewards[self.agents[1]]
#         rewards[self.agents[1]] += (robot_com_pos_final[1][0] - robot_com_pos_init[1][0]) * 0.5
#         rewards[self.agents[0]] -= (robot_com_pos_final[0][0] - robot_com_pos_init[0][0]) * 0.5

#         # negative reward for robot/package separating
#         rewards[self.agents[1]] += abs(robot_com_pos_init[1][0] - package_com_pos_init[0]) - abs(
#             robot_com_pos_final[1][0] - package_com_pos_final[0]
#         )
#         rewards[self.agents[0]] += abs(robot_com_pos_init[0][0] - package_com_pos_init[0]) - abs(
#             robot_com_pos_final[0][0] - package_com_pos_final[0]
#         )

#         return rewards

#     def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

#         assert self.timestep is not None, "You must call reset before calling step"

#         # collect pre step information
#         package_pos_init = self.object_pos_at_time(self.get_time(), "package")
#         robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#         package_com_pos_init = np.mean(package_pos_init, axis=1)
#         robot_com_pos_init = [np.mean(pos, axis=1) for pos in robot_pos_init]

#         # step
#         is_unstable = super(MultiAgentEvoGymBase, self).step(action)

#         # collect post step information
#         package_pos_final = self.object_pos_at_time(self.get_time(), "package")
#         robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
#         package_com_pos_final = np.mean(package_pos_final, axis=1)
#         robot_com_pos_final = [np.mean(pos, axis=1) for pos in robot_pos_final]

#         # calculate reward
#         rewards = self.get_rewards(
#             package_com_pos_init,
#             package_com_pos_final,
#             robot_com_pos_init,
#             robot_com_pos_final,
#         )

#         # judge termination
#         left_most = np.min(package_pos_final[0])
#         right_most = np.max(package_pos_final[0])
#         terminations = {a: True for a in self.agents}

#         if left_most < self.LEFT_THRESH:
#             rewards[self.agents[0]] += self.COMPLETION_REWARD
#             rewards[self.agents[1]] -= self.COMPLETION_REWARD
#         elif right_most > self.RIGHT_THRESH:
#             rewards[self.agents[0]] -= self.COMPLETION_REWARD
#             rewards[self.agents[1]] += self.COMPLETION_REWARD
#         elif is_unstable:
#             print("SIMULATION UNSTABLE... TERMINATING")
#             for a in self.agents:
#                 rewards[a] -= self.COMPLETION_REWARD
#         else:
#             terminations = {a: False for a in self.agents}

#         # judge truncation
#         if self.timestep >= 1000:
#             rewards = {a: 0.0 for a in self.agents}
#             truncations = {a: True for a in self.agents}
#         else:
#             truncations = {a: False for a in self.agents}
#         self.timestep += 1

#         # return
#         observations = self.calc_obs(robot_com_pos_final, package_com_pos_final)
#         infos: InfoDict = {a: {} for a in self.agents}

#         if all(terminations.values()) or all(truncations.values()):
#             self.agents = []

#         return observations, rewards, terminations, truncations, infos


class WallPushEnvClass(PackageBase):

    ENV_NAME = "WallPusher-v0"
    ENV_FILE_NAME = "WallPusher-v0.json"
    ROBOT1_INIT_POS = (19, 1)
    ROBOT2_INIT_POS = (27, 1)

    VIEWER_DEFAULT_POS = (17.5, 4)

    LEFT_THRESH = 1 * MultiAgentEvoGymBase.VOXEL_SIZE
    RIGHT_THRESH = 50 * MultiAgentEvoGymBase.VOXEL_SIZE
    COMPLETION_REWARD = 1.0

    def get_rewards(
        self,
        package_com_pos_init: np.ndarray,
        package_com_pos_final: np.ndarray,
        robot_com_pos_init: List[np.ndarray],
        robot_com_pos_final: List[np.ndarray],
    ) -> RewardDict:

        rewards = {a: 0.0 for a in self.agents}

        # positive reward for moving backward
        rewards[self.agents[1]] += (package_com_pos_final[0] - package_com_pos_init[0]) * 0.75
        rewards[self.agents[0]] -= rewards[self.agents[1]]
        rewards[self.agents[1]] += (robot_com_pos_final[1][0] - robot_com_pos_init[1][0]) * 0.5
        rewards[self.agents[0]] -= (robot_com_pos_final[0][0] - robot_com_pos_init[0][0]) * 0.5

        return rewards

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        assert self.timestep is not None, "You must call reset before calling step"

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        package_com_pos_init = np.mean(package_pos_init, axis=1)
        robot_com_pos_init = [np.mean(pos, axis=1) for pos in robot_pos_init]

        # step
        is_unstable = super(MultiAgentEvoGymBase, self).step(action)

        # collect post step information
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        package_com_pos_final = np.mean(package_pos_final, axis=1)
        robot_com_pos_final = [np.mean(pos, axis=1) for pos in robot_pos_final]

        # calculate reward
        rewards = self.get_rewards(
            package_com_pos_init,
            package_com_pos_final,
            robot_com_pos_init,
            robot_com_pos_final,
        )

        # judge termination
        left_most = np.min(package_pos_final[0])
        right_most = np.max(package_pos_final[0])
        terminations = {a: True for a in self.agents}

        if left_most < self.LEFT_THRESH:
            rewards[self.agents[0]] += self.COMPLETION_REWARD
            rewards[self.agents[1]] -= self.COMPLETION_REWARD
        elif right_most > self.RIGHT_THRESH:
            rewards[self.agents[0]] -= self.COMPLETION_REWARD
            rewards[self.agents[1]] += self.COMPLETION_REWARD
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
        observations = self.calc_obs(robot_com_pos_final, package_com_pos_final)
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos
