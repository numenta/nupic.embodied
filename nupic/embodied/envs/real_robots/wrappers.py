# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------
from gym import spaces


class WrapRobot(object):
    """
    Wrapper for the REALRobotEnv (based on MJCFBaseBulletEnv).
    Returns only Box of retina image as observation instead of
    Dictionary of (joint_positions, touch_sensors, retina, depth,
    mask, object_positions, goal, goal_mask, goal_positions).
    This is required for the openAI stable-baselines algorithms
    as they don't support Dict observations.
    Also the action space is reformated into the Box format, omitting
    the render parameter.

    crop_obs: Specifies whether the retina image should be cropped from
             240x320x3 to 180x180x3 (mainly cuts off white space)
    """

    def __init__(self, env, crop_obs=False):
        self._env = env
        self.crop_obs = crop_obs

        self._env.action_space = self._env.action_space["joint_command"]
        if self.crop_obs:
            self._env.observation_space = spaces.Box(
                low=0, high=255.0, shape=(180, 180, 3), dtype="float32"
            )
        else:
            self._env.observation_space = self._env.observation_space["retina"]

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        action = {"joint_command": action, "render": True}
        observation, reward, done, info = self._env.step_joints(action)
        if self.crop_obs:
            observation = observation["retina"][0:180, 70:250, :]
        else:
            observation = observation["retina"]
        return observation, reward, done, info

    def reset(self):
        observation = self._env.reset()
        if self.crop_obs:
            observation = observation["retina"][0:180, 70:250, :]
        else:
            observation = observation["retina"]
        return observation


class GoalWrapper(object):
    """
    Wrapper for the REALRobotEnv (based on MJCFBaseBulletEnv).
    Returns Dict of 3 Box items (observation, achieved_goal,
    desired_goal) as observation instead of the entire
    Dictionary of (joint_positions, touch_sensors, retina, depth,
    mask, object_positions, goal, goal_mask, goal_positions).
    This is required follows the naming and format of the openAI
    stable-baselines her implementation.
    For observation the retina image is used, the two goal elements
    are currently placeholder.
    The stable_baselines3 implementation currently only supports 1D
    observations which is why the images are flattened.
    Also the action space is reformated into the Box format, omitting
    the render parameter.

    crop_obs: Specifies whether the retina image + desired & achieved
            goal should be cropped from 240x320x3 to 180x180x3 (mainly
            cuts off white space)
    """

    def __init__(self, env, crop_obs=False):
        self._env = env
        self.crop_obs = crop_obs

        if self.crop_obs:
            obs_shape_1d = 180 * 180 * 3
        else:
            obs_shape = env.observation_space["retina"].shape
            obs_shape_1d = obs_shape[0] * obs_shape[1] * obs_shape[2]

        self._env.action_space = self._env.action_space["joint_command"]
        # TODO: replace goal placeholders with actual goals
        self._env.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    low=0, high=255.0, shape=[obs_shape_1d], dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    low=0, high=255.0, shape=[obs_shape_1d], dtype="float32"
                ),
                observation=spaces.Box(
                    low=0, high=255.0, shape=[obs_shape_1d], dtype="float32"
                ),
            )
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        action = {"joint_command": action, "render": True}
        observation, reward, done, info = self._env.step_joints(action)
        if self.crop_obs:
            observation = {
                "observation": observation["retina"][0:180, 70:250, :].flatten(),
                "achieved_goal": observation["goal"][0:180, 70:250, :].flatten(),
                "desired_goal": observation["goal"][0:180, 70:250, :].flatten(),
            }
        else:
            observation = {
                "observation": observation["retina"].flatten(),
                "achieved_goal": observation["goal"].flatten(),
                "desired_goal": observation["goal"].flatten(),
            }
        return observation, reward, done, info

    def reset(self):
        observation = self._env.reset()

        if self.crop_obs:
            observation = {
                "observation": observation["retina"][0:180, 70:250, :].flatten(),
                "achieved_goal": observation["goal"][0:180, 70:250, :].flatten(),
                "desired_goal": observation["goal"][0:180, 70:250, :].flatten(),
            }
        else:
            observation = {
                "observation": observation["retina"].flatten(),
                "achieved_goal": observation["goal"].flatten(),
                "desired_goal": observation["goal"].flatten(),
            }

        return observation
