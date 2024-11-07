import os

import numpy as np
from evogym import EvoWorld  # type:ignore
from evogym.envs import EvoGymBase  # type:ignore
from gym import spaces  # type:ignore


class SimpleSumoEnvClass(EvoGymBase):
    def __init__(self, structure_1, structure_2):

        # parse structures
        body_1, connections_1 = structure_1
        body_2, connections_2 = structure_2

        # make world
        self.world = EvoWorld.from_json(
            os.path.join("world_data", "simple_sumo_env.json")
        )
        self.world.add_from_array(
            "robot_1", body_1, 15 - body_1.shape[1], 1, connections=connections_1
        )
        self.world.add_from_array("robot_2", body_2, 16, 1, connections=connections_2)

        # init sim
        EvoGymBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators_1 = self.get_actuator_indices("robot_1").size
        num_actuators_2 = self.get_actuator_indices("robot_2").size

        num_robot_points_1 = self.object_pos_at_time(self.get_time(), "robot_1").size
        num_robot_points_2 = self.object_pos_at_time(self.get_time(), "robot_2").size

        self.action_space = spaces.Dict(
            {
                "robot_1": spaces.Box(
                    low=0.6, high=1.6, shape=(num_actuators_1,), dtype=np.float32
                ),
                "robot_2": spaces.Box(
                    low=0.6, high=1.6, shape=(num_actuators_2,), dtype=np.float32
                ),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "robot_1": spaces.Box(
                    low=-100.0,
                    high=100.0,
                    shape=(6 + num_robot_points_1,),
                    dtype=np.float32,
                ),
                "robot_2": spaces.Box(
                    low=-100.0,
                    high=100.0,
                    shape=(6 + num_robot_points_2,),
                    dtype=np.float32,
                ),
            }
        )

        # set viewer
        self.default_viewer.track_objects("robot_1", "robot_2")

    def step(self, action):

        # collect pre step information
        robot_1_pos_init = self.object_pos_at_time(self.get_time(), "robot_1")
        robot_2_pos_init = self.object_pos_at_time(self.get_time(), "robot_2")

        # When this is True, the simulation has reached an unstable state from which it cannot recover
        done = super().step(action)

        # collect post step information
        robot_1_pos_final = self.object_pos_at_time(self.get_time(), "robot_1")
        robot_1_vel_final = self.object_vel_at_time(self.get_time(), "robot_1")
        robot_2_pos_final = self.object_pos_at_time(self.get_time(), "robot_2")
        robot_2_vel_final = self.object_vel_at_time(self.get_time(), "robot_2")

        # calculate positions and velocities of center of mass
        robot_1_com_pos_init = np.mean(robot_1_pos_init, 1)
        robot_1_com_pos_final = np.mean(robot_1_pos_final, 1)
        robot_1_com_vel_final = np.mean(robot_1_vel_final, 1)
        robot_2_com_pos_init = np.mean(robot_2_pos_init, 1)
        robot_2_com_pos_final = np.mean(robot_2_pos_final, 1)
        robot_2_com_vel_final = np.mean(robot_2_vel_final, 1)

        # calculate reward
        reward_1 = robot_1_com_pos_final[0] - robot_1_com_pos_init[0]
        reward_2 = -(robot_2_com_pos_final[0] - robot_2_com_pos_init[0])
        reward = {"robot_1": reward_1, "robot_2": reward_2}

        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
        if robot_1_com_pos_final[0] > 28:
            done = True
            reward_1 += 1.0
            reward_2 -= 1.0
        if robot_2_com_pos_final[0] < 2:
            done = True
            reward_1 -= 1.0
            reward_2 += 1.0

        # calulate observations
        robots_distance_x = robot_2_com_pos_final[0] - robot_1_com_pos_final[0]
        robots_distance_y = robot_2_com_pos_final[1] - robot_1_com_pos_final[1]

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        robot_1_com_vel_final[0],
                        robot_1_com_vel_final[1],
                        robot_2_com_vel_final[0],
                        robot_2_com_vel_final[1],
                        robots_distance_x,
                        robots_distance_y,
                    ]
                ),
                self.get_relative_pos_obs("robot_1"),
            )
        )

        obs2 = np.concatenate(
            (
                np.array(
                    [
                        robot_2_com_vel_final[0],
                        robot_2_com_vel_final[1],
                        robot_1_com_vel_final[0],
                        robot_1_com_vel_final[1],
                        robots_distance_x,
                        robots_distance_y,
                    ]
                ),
                self.get_relative_pos_obs("robot_2"),
            )
        )

        obs = {"robot_1": obs1, "robot_2": obs2}

        return obs, reward, done, {}

    def reset(self):

        super().reset()

        # collect post step information
        robot_1_pos_final = self.object_pos_at_time(self.get_time(), "robot_1")
        robot_1_vel_final = self.object_vel_at_time(self.get_time(), "robot_1")
        robot_2_pos_final = self.object_pos_at_time(self.get_time(), "robot_2")
        robot_2_vel_final = self.object_vel_at_time(self.get_time(), "robot_2")

        # calculate positions and velocities of center of mass
        robot_1_com_pos_final = np.mean(robot_1_pos_final, 1)
        robot_1_com_vel_final = np.mean(robot_1_vel_final, 1)
        robot_2_com_pos_final = np.mean(robot_2_pos_final, 1)
        robot_2_com_vel_final = np.mean(robot_2_vel_final, 1)

        # calulate observations
        robots_distance_x = robot_2_com_pos_final[0] - robot_1_com_pos_final[0]
        robots_distance_y = robot_2_com_pos_final[1] - robot_1_com_pos_final[1]

        obs1 = np.concatenate(
            (
                np.array(
                    [
                        robot_1_com_vel_final[0],
                        robot_1_com_vel_final[1],
                        robot_2_com_vel_final[0],
                        robot_2_com_vel_final[1],
                        robots_distance_x,
                        robots_distance_y,
                    ]
                ),
                self.get_relative_pos_obs("robot_1"),
            )
        )

        obs2 = np.concatenate(
            (
                np.array(
                    [
                        robot_2_com_vel_final[0],
                        robot_2_com_vel_final[1],
                        robot_1_com_vel_final[0],
                        robot_1_com_vel_final[1],
                        robots_distance_x,
                        robots_distance_y,
                    ]
                ),
                self.get_relative_pos_obs("robot_2"),
            )
        )

        obs = {"robot_1": obs1, "robot_2": obs2}
        return obs
