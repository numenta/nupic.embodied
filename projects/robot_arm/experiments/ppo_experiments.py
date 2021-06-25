#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
ppo through rewards experiments.
"""

from copy import deepcopy

from .base import ppo_base

from nupic.embodied.policies.dynamics import Dynamics

from nupic.embodied.policies.curious_cnn_policy import CnnPolicy

# On hooks: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/


class CustomDynamics(Dynamics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"In {self.__class__}")
        for name, module in self.dynamics_net.named_modules():
            module.register_backward_hook(backward_hook)


class CnnPolicyWithBackwardHooks(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"In {self.__class__}")
        for name, module in self.pd_hidden.named_modules():
            module.register_backward_hook(backward_hook)


def backward_hook(m, i, o):
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape)
        except AttributeError:
            print("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:
        try:
            print(grad.shape)
        except AttributeError:
            print("None found for Gradient")
    print("\n")


# Debug experiment only
ppo_debug_robot = deepcopy(ppo_base)
ppo_debug_robot.update(
    envs_per_process=8,
    num_timesteps=10000,  # 100000 actual number of steps should be higher
    env_kind="roboarm",
    env="RRA",
    touch_reward=True,
    random_force=True,
    ppo_through_reward=False,
    group="Testppo",
    notes="Testing ppo experiments",
    use_disagreement=True,  # currently not working when set to False
    dynamics_class=CustomDynamics,
    policy_class=CnnPolicyWithBackwardHooks,
)

# Debug PPO in atari breakout
ppo_debug_atari = deepcopy(ppo_base)
ppo_debug_atari.update(
    envs_per_process=8,
    ppo_through_reward=False,
)

# Export configurations in this file
CONFIGS = dict(
    ppo_debug_robot=ppo_debug_robot,
    ppo_debug_atari=ppo_debug_atari,
    # ppo_debug_atari=ppo_debug_atari,
)
