from copy import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.base import MultiAgentEvoGymBase
from envs.typehints import ActionDict, BoolDict, InfoDict, ObsDict, RewardDict


class OjamaDepth4EnvClass(MultiAgentEvoGymBase):

    ENV_NAME = "Ojama-d4"
    ADDITIONAL_OBS_DIM = 17
    X_THRESH = 21 * MultiAgentEvoGymBase.VOXEL_SIZE
    COMPLETION_REWARD = 3.0

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
        env_file_name = "Ojama-depth-4.json"
        x_positions = [8, 15]
        y_positions = [6, 2]

        super().__init__(
            body_list,
            connections_list,
            env_file_name,
            render_mode,
            render_options,
            x_positions,
            y_positions,
        )

        self.default_viewer.track_objects(*self.possible_agents)

    def step(self, actions: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]
        robot_2_min_pos_init = np.min(robot_pos_init[1], axis=1)
        robot_2_max_pos_init = np.max(robot_pos_init[1], axis=1)

        assert self.timestep is not None, "Timestep is None. Did you call reset()?"

        is_unstable = super(MultiAgentEvoGymBase, self).step(actions)

        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]
        robot_2_min_pos_final = np.min(robot_pos_final[1], axis=1)
        robot_2_max_pos_final = np.max(robot_pos_final[1], axis=1)

        span_final = robot_2_max_pos_final[1] - robot_2_min_pos_final[1]
        span_init = robot_2_max_pos_init[1] - robot_2_min_pos_init[1]

        rewards = {a: 0.0 for a in self.agents}

        rewards[self.agents[0]] += robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
        rewards[self.agents[1]] += span_final - span_init
        # rewards[self.agents[1]] += robot_com_pos_final[1][1] - robot_com_pos_init[1][1]
        # rewards[self.agents[1]] += np.max(robot_pos_final[1], axis=1)[1] - np.max(robot_pos_init[1], axis=1)[1]
        # rewards[self.agents[1]] -= abs(robot_com_pos_final[1][0] - robot_com_pos_init[1][0]) * 0.5

        terminations = {a: True for a in self.agents}

        if np.min(robot_pos_final[0], axis=1)[0] >= self.X_THRESH:
            rewards[self.agents[0]] += self.COMPLETION_REWARD
            rewards[self.agents[1]] -= self.COMPLETION_REWARD
        elif is_unstable:
            print("SIMULATION UNSTABLE ... TERMINATING")
            rewards[self.agents[0]] -= self.COMPLETION_REWARD
            rewards[self.agents[1]] -= self.COMPLETION_REWARD
        else:
            terminations = {a: False for a in self.agents}

        if self.timestep >= 600:
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}
        self.timestep += 1

        observations = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
        obs = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        return obs, infos

    def calc_obs(self) -> ObsDict:

        robot_com_pos = [self.get_pos_com_obs(a) for a in self.agents]
        x_distance = robot_com_pos[0][0] - robot_com_pos[1][0]
        y_distance = robot_com_pos[0][1] - robot_com_pos[1][1]

        robot_orientations = [self.object_orientation_at_time(self.get_time(), a) for a in self.agents]

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        x_distance,
                        y_distance,
                        robot_orientations[0],
                        robot_orientations[1],
                    ]
                ),
                self.get_vel_com_obs(self.agents[0]),
                self.get_relative_pos_obs(self.agents[0]),
                self.get_floor_obs(self.agents[0], ["ground", self.agents[1]], 5),
            )
        )

        obs2 = np.concatenate(
            (
                np.array(
                    [
                        x_distance,
                        y_distance,
                        robot_orientations[0],
                        robot_orientations[1],
                    ]
                ),
                self.get_vel_com_obs(self.agents[1]),
                self.get_relative_pos_obs(self.agents[1]),
                self.get_floor_obs(self.agents[1], ["ground"], 5),
            )
        )

        observations = {
            self.agents[0]: obs1,
            self.agents[1]: obs2,
        }

        return observations


class OjamaDepth3EnvClass(MultiAgentEvoGymBase):

    ENV_NAME = "Ojama-d3"
    ADDITIONAL_OBS_DIM = 17
    X_THRESH = 21 * MultiAgentEvoGymBase.VOXEL_SIZE
    COMPLETION_REWARD = 3.0

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
        env_file_name = "Ojama-depth-3.json"
        x_positions = [8, 15]
        y_positions = [6, 3]

        super().__init__(
            body_list,
            connections_list,
            env_file_name,
            render_mode,
            render_options,
            x_positions,
            y_positions,
        )

        self.default_viewer.track_objects(*self.possible_agents)

    def step(self, actions: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]
        robot_2_min_pos_init = np.min(robot_pos_init[1], axis=1)
        robot_2_max_pos_init = np.max(robot_pos_init[1], axis=1)

        assert self.timestep is not None, "Timestep is None. Did you call reset()?"

        is_unstable = super(MultiAgentEvoGymBase, self).step(actions)

        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]
        robot_2_min_pos_final = np.min(robot_pos_final[1], axis=1)
        robot_2_max_pos_final = np.max(robot_pos_final[1], axis=1)

        span_final = robot_2_max_pos_final[1] - robot_2_min_pos_final[1]
        span_init = robot_2_max_pos_init[1] - robot_2_min_pos_init[1]

        rewards = {a: 0.0 for a in self.agents}

        rewards[self.agents[0]] += robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
        rewards[self.agents[1]] += span_final - span_init
        # rewards[self.agents[1]] += robot_com_pos_final[1][1] - robot_com_pos_init[1][1]
        # rewards[self.agents[1]] += np.max(robot_pos_final[1], axis=1)[1] - np.max(robot_pos_init[1], axis=1)[1]
        # rewards[self.agents[1]] -= abs(robot_com_pos_final[1][0] - robot_com_pos_init[1][0]) * 0.5

        terminations = {a: True for a in self.agents}

        if np.min(robot_pos_final[0], axis=1)[0] >= self.X_THRESH:
            rewards[self.agents[0]] += self.COMPLETION_REWARD
            rewards[self.agents[1]] -= self.COMPLETION_REWARD
        elif is_unstable:
            print("SIMULATION UNSTABLE ... TERMINATING")
            rewards[self.agents[0]] -= self.COMPLETION_REWARD
            rewards[self.agents[1]] -= self.COMPLETION_REWARD
        else:
            terminations = {a: False for a in self.agents}

        if self.timestep >= 600:
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}
        self.timestep += 1

        observations = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
        obs = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        return obs, infos

    def calc_obs(self) -> ObsDict:

        robot_com_pos = [self.get_pos_com_obs(a) for a in self.agents]
        x_distance = robot_com_pos[0][0] - robot_com_pos[1][0]
        y_distance = robot_com_pos[0][1] - robot_com_pos[1][1]

        robot_orientations = [self.object_orientation_at_time(self.get_time(), a) for a in self.agents]

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        x_distance,
                        y_distance,
                        robot_orientations[0],
                        robot_orientations[1],
                    ]
                ),
                self.get_vel_com_obs(self.agents[0]),
                self.get_relative_pos_obs(self.agents[0]),
                self.get_floor_obs(self.agents[0], ["ground", self.agents[1]], 5),
            )
        )

        obs2 = np.concatenate(
            (
                np.array(
                    [
                        x_distance,
                        y_distance,
                        robot_orientations[0],
                        robot_orientations[1],
                    ]
                ),
                self.get_vel_com_obs(self.agents[1]),
                self.get_relative_pos_obs(self.agents[1]),
                self.get_floor_obs(self.agents[1], ["ground"], 5),
            )
        )

        observations = {
            self.agents[0]: obs1,
            self.agents[1]: obs2,
        }

        return observations
    

class OjamaDepth5EnvClass(MultiAgentEvoGymBase):

    ENV_NAME = "Ojama-d5"
    ADDITIONAL_OBS_DIM = 17
    X_THRESH = 21 * MultiAgentEvoGymBase.VOXEL_SIZE
    COMPLETION_REWARD = 3.0

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
        env_file_name = "Ojama-depth-5.json"
        x_positions = [8, 15]
        y_positions = [6, 1]

        super().__init__(
            body_list,
            connections_list,
            env_file_name,
            render_mode,
            render_options,
            x_positions,
            y_positions,
        )

        self.default_viewer.track_objects(*self.possible_agents)

    def step(self, actions: ActionDict) -> Tuple[ObsDict, RewardDict, BoolDict, BoolDict, InfoDict]:

        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]
        robot_2_min_pos_init = np.min(robot_pos_init[1], axis=1)
        robot_2_max_pos_init = np.max(robot_pos_init[1], axis=1)

        assert self.timestep is not None, "Timestep is None. Did you call reset()?"

        is_unstable = super(MultiAgentEvoGymBase, self).step(actions)

        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in self.agents]
        robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]
        robot_2_min_pos_final = np.min(robot_pos_final[1], axis=1)
        robot_2_max_pos_final = np.max(robot_pos_final[1], axis=1)

        span_final = robot_2_max_pos_final[1] - robot_2_min_pos_final[1]
        span_init = robot_2_max_pos_init[1] - robot_2_min_pos_init[1]

        rewards = {a: 0.0 for a in self.agents}

        rewards[self.agents[0]] += robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
        rewards[self.agents[1]] += span_final - span_init
        # rewards[self.agents[1]] += robot_com_pos_final[1][1] - robot_com_pos_init[1][1]
        # rewards[self.agents[1]] += np.max(robot_pos_final[1], axis=1)[1] - np.max(robot_pos_init[1], axis=1)[1]
        # rewards[self.agents[1]] -= abs(robot_com_pos_final[1][0] - robot_com_pos_init[1][0]) * 0.5

        terminations = {a: True for a in self.agents}

        if np.min(robot_pos_final[0], axis=1)[0] >= self.X_THRESH:
            rewards[self.agents[0]] += self.COMPLETION_REWARD
            rewards[self.agents[1]] -= self.COMPLETION_REWARD
        elif is_unstable:
            print("SIMULATION UNSTABLE ... TERMINATING")
            rewards[self.agents[0]] -= self.COMPLETION_REWARD
            rewards[self.agents[1]] -= self.COMPLETION_REWARD
        else:
            terminations = {a: False for a in self.agents}

        if self.timestep >= 600:
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}
        self.timestep += 1

        observations = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsDict, InfoDict]:

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        super(MultiAgentEvoGymBase, self).reset(seed=seed, options=options)
        obs = self.calc_obs()
        infos: InfoDict = {a: {} for a in self.agents}

        return obs, infos

    def calc_obs(self) -> ObsDict:

        robot_com_pos = [self.get_pos_com_obs(a) for a in self.agents]
        x_distance = robot_com_pos[0][0] - robot_com_pos[1][0]
        y_distance = robot_com_pos[0][1] - robot_com_pos[1][1]

        robot_orientations = [self.object_orientation_at_time(self.get_time(), a) for a in self.agents]

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        x_distance,
                        y_distance,
                        robot_orientations[0],
                        robot_orientations[1],
                    ]
                ),
                self.get_vel_com_obs(self.agents[0]),
                self.get_relative_pos_obs(self.agents[0]),
                self.get_floor_obs(self.agents[0], ["ground", self.agents[1]], 5),
            )
        )

        obs2 = np.concatenate(
            (
                np.array(
                    [
                        x_distance,
                        y_distance,
                        robot_orientations[0],
                        robot_orientations[1],
                    ]
                ),
                self.get_vel_com_obs(self.agents[1]),
                self.get_relative_pos_obs(self.agents[1]),
                self.get_floor_obs(self.agents[1], ["ground"], 5),
            )
        )

        observations = {
            self.agents[0]: obs1,
            self.agents[1]: obs2,
        }

        return observations