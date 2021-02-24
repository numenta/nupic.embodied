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
import real_robots.evaluate
from nupic.embodied.envs.real_robots import RandomPolicy

# explanation of variables here: https://github.com/emilio-cartoni/REAL2020_starter_kit
# set visualize to True to watch :)

# Test random policy
result, detailed_scores = real_robots.evaluate(
                RandomPolicy,
                environment='R1',
                action_type='joints',
                n_objects=3,
                intrinsic_timesteps=1e3,
                extrinsic_timesteps=1e3,
                extrinsic_trials=3,
                visualize=False,
                goals_dataset_path='./goals/goals-REAL2020-sNone-25-15-10-3.npy.npz'
            )
# NOTE : You can find goals-REAL2020-s2020-50-1.npy.npz file in the REAL2020 Starter Kit repository
# or you can generate one using the real-robots-generate-goals command.
#
print(result)
# {'score_REAL2020': 0.06529471503519801, 'score_total': 0.06529471503519801}
print(detailed_scores)
# {'REAL2020': [0.00024387094790936833, 0.19553060745741896, 0.00010966670026571288]}
