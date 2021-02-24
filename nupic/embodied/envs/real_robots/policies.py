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
from real_robots.policy import BasePolicy
import numpy as np

"""
TODO::
add the start_intrinsic_phase, end_intrinsic_phase, start_extrinsic_phase,
end_extrinsic_phase, start_extrinsic_trial, end_extrinsic_trial functions
from https://github.com/AIcrowd/real_robots/blob/6dd5b70bad14426483e2d3ee29b3d8708d34e1ba/real_robots/policy.py
to perform actions at start or end of phases and trials.
"""


class RandomPolicy(BasePolicy):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.action = action_space.sample()

    def step(self, observation, reward, done):
        if np.random.rand() < 0.05:
            self.action = self.action_space.sample()
        return self.action
