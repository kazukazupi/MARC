import os
from typing import Any, Dict, List, Optional

import numpy as np
from evogym import EvoWorld  # type:ignore
from evogym.envs import EvoGymBase  # type:ignore
from gymnasium import spaces  # type:ignore

ROBOT_1 = "robot_1"
ROBOT_2 = "robot_2"


class SimpleSumoEnvClass(EvoGymBase):
    def __init__(
        self,
        structure_1,
        structure_2,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        # parse structures
        body_1, connections_1 = structure_1
        body_2, connections_2 = structure_2

        # make world
        self.world = EvoWorld.from_json(os.path.join("world_data", "simple_sumo_env.json"))
        self.world.add_from_array(ROBOT_1, body_1, 15 - body_1.shape[1], 1, connections=connections_1)
        self.world.add_from_array(ROBOT_2, body_2, 16, 1, connections=connections_2)

        # init sim
        EvoGymBase.__init__(self, self.world, render_mode, render_options)

        # set action space and observation space
        num_actuators = [self.get_actuator_indices(obj).size for obj in [ROBOT_1, ROBOT_2]]

        num_robot_points = [self.object_pos_at_time(self.get_time(), obj).size for obj in [ROBOT_1, ROBOT_2]]

        self.action_space = spaces.Dict(
            {
                obj: spaces.Box(low=0.6, high=1.6, shape=(num_actuators[i],), dtype=float)
                for i, obj in enumerate([ROBOT_1, ROBOT_2])
            }
        )

        self.observation_space = spaces.Dict(
            {
                obj: spaces.Box(
                    low=-100.0,
                    high=100.0,
                    shape=(6 + num_robot_points[i],),
                    dtype=float,
                )
                for i, obj in enumerate([ROBOT_1, ROBOT_2])
            }
        )

        # set viewer
        self.default_viewer.track_objects(ROBOT_1, ROBOT_2)

    def step(self, action):

        # collect pre step information
        robot_pos_init = [self.object_pos_at_time(self.get_time(), obj) for obj in [ROBOT_1, ROBOT_2]]

        # When this is True, the simulation has reached an unstable state from which it cannot recover
        done = super().step(action)

        # collect post step information
        robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in [ROBOT_1, ROBOT_2]]

        # calculate positions and velocities of center of mass
        robot_com_pos_init = [np.mean(pos, 1) for pos in robot_pos_init]
        robot_com_pos_final = [np.mean(pos, 1) for pos in robot_pos_final]

        # calculate reward
        reward_1 = robot_com_pos_final[0][0] - robot_com_pos_init[0][0]
        reward_2 = -(robot_com_pos_final[1][0] - robot_com_pos_init[1][0])

        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
        if robot_com_pos_final[0][0] > 28:
            done = True
            reward_1 += 1.0
            reward_2 -= 1.0
        if robot_com_pos_final[1][0] < 2:
            done = True
            reward_1 -= 1.0
            reward_2 += 1.0

        reward = {ROBOT_1: reward_1, ROBOT_2: reward_2}

        obs = self.calc_obs(robot_pos_final, robot_com_pos_final)

        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):

        super().reset(seed=seed, options=options)
        obs = self.calc_obs()

        return obs, {}

    def calc_obs(
        self,
        robot_pos_final: Optional[List[np.ndarray]] = None,
        robot_com_pos_final: Optional[List[np.ndarray]] = None,
    ):

        # collect post step information
        if robot_pos_final is None:
            robot_pos_final = [self.object_pos_at_time(self.get_time(), obj) for obj in [ROBOT_1, ROBOT_2]]

        robot_vel_final = [self.object_vel_at_time(self.get_time(), obj) for obj in [ROBOT_1, ROBOT_2]]

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
                self.get_relative_pos_obs(ROBOT_1),
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
                self.get_relative_pos_obs(ROBOT_2),
            )
        )

        obs = {ROBOT_1: obs1, ROBOT_2: obs2}

        return obs
