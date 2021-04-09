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
"""This modules creates a continuous Q-function network with a Dendrite MLP."""

import torch

from nupic.embodied.models import DendriticMLP


class ContinuousDendriteMLPQFunction(DendriticMLP):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, dim_context, **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        self._dim_context = dim_context
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim - self._dim_context
        self._action_dim = env_spec.action_space.flat_dim

        DendriticMLP.__init__(self,
                              input_size=self._obs_dim + self._action_dim,
                              output_dim=1,
                              dim_context=dim_context,
                              **kwargs)

    def forward(self, observations, actions):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.

        Returns:
            torch.Tensor: Output value
        """
        obs_portion = observations[:, :self._obs_dim]
        context_portion = observations[:, self._obs_dim:]
        return super().forward(torch.cat([obs_portion, actions], 1), context_portion)
