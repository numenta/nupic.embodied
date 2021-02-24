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
import gym
import numpy as np
import time
from nupic.embodied.envs.real_robots import (
    RandomPolicy,
    WrapRobot,
)
from real_robots.envs import REALRobotEnv

import torch

from stable_baselines3 import PPO


print("setting up environment")
#env = gym.make("REALRobot2020-R2J3-v0")
env = REALRobotEnv(objects=1)
env = WrapRobot(env, crop_obs=True)

print("setting up ppo model")
# stable_baselines3 "registers policy" so they can be passed as string argument
# CnnPolicy not compatible with box format, expects image
# model = PPO("CnnPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1)
print("start learning")
model.learn(total_timesteps=10)  # 256
print("learning done")

#Here we need to restart the environent to make rendering possible
env = REALRobotEnv(objects=1)
env = WrapRobot(env, crop_obs=True)

env.render("human")

print("display model")
observation = env.reset()
action = env.action_space.sample()
reward, done = 0, False
for t in range(400):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)
