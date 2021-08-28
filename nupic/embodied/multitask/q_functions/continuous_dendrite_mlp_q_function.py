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

from nupic.research.frameworks.dendrites.modules import AbsoluteMaxGatingDendriticLayer
from nupic.embodied.multitask.models.dendrite_mlp import CustomDendriticMLP


class ContinuousDendriteMLPQFunction(CustomDendriticMLP):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes,
                 num_segments,
                 dim_context,
                 kw,
                 kw_percent_on=0.05,
                 context_percent_on=1.0,
                 weight_sparsity=0.95,
                 weight_init="modified",
                 dendrite_init="modified",
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer,
                 output_nonlinearity=None,
                 preprocess_module_type=None,
                 preprocess_output_dim=128,
                 representation_module_type=None,
                 representation_module_dims=(128, 128)):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        self._dim_context = dim_context
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim - self._dim_context
        self._action_dim = env_spec.action_space.flat_dim

        CustomDendriticMLP.__init__(self,
                                    input_dim=self._obs_dim + self._action_dim,
                                    output_sizes=1,
                                    dim_context=dim_context,
                                    hidden_sizes=hidden_sizes,
                                    num_segments=num_segments,
                                    kw=kw,
                                    kw_percent_on=kw_percent_on,
                                    context_percent_on=context_percent_on,
                                    weight_sparsity=weight_sparsity,
                                    weight_init=weight_init,
                                    dendrite_init=dendrite_init,
                                    dendritic_layer_class=dendritic_layer_class,
                                    output_nonlinearity=output_nonlinearity,
                                    preprocess_module_type=preprocess_module_type,
                                    preprocess_output_dim=preprocess_output_dim,
                                    preprocess_kw_percent_on=kw_percent_on,
                                    representation_module_type=representation_module_type,
                                    representation_module_dims=representation_module_dims
                                    )

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
