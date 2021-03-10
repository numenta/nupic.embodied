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

import unittest

from gym import spaces
from real_robots.envs import REALRobotEnv

from nupic.embodied.envs.real_robots import WrapRobot


class WrapRobotTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_observation_space_is_box_when_cropping(self):
        """
        Test if the observation space is of type spaces.Box when cropping
        """
        env = REALRobotEnv(objects=1)
        env = WrapRobot(env, crop_obs=True)
        self.assertIsInstance(env.observation_space, spaces.Box)


if __name__ == "__main__":
    unittest.main(verbosity=2)
