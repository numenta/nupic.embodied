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

from nupic.embodied.multitask.modules import CustomDendriticMLP

class ContinuousDendriteMLPQFunction(CustomDendriticMLP):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(
        self,
        env_spec,
        num_tasks,
        input_data, 
        context_data,
        hidden_sizes,
        layers_modulated,
        num_segments,
        kw_percent_on,
        context_percent_on,
        weight_sparsity,
        weight_init,
        dendrite_weight_sparsity,
        dendrite_init,
        dendritic_layer_class,
        output_nonlinearity,
        preprocess_module_type,
        preprocess_output_dim
    ):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        self.num_tasks = num_tasks

        self.input_data = input_data
        self.context_data = context_data

        if input_data == "obs":
            self.input_dim = env_spec.observation_space.flat_dim - self.num_tasks + env_spec.action_space.flat_dim
        elif input_data == "obs|context":
            self.input_dim = env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim
        
        if context_data == "context":
            self.context_dim = self.num_tasks
        elif context_data == "obs|context":
            self.context_dim = env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim


        super().__init__(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_sizes=1,
            hidden_sizes=hidden_sizes,
            layers_modulated=layers_modulated,
            num_segments=num_segments,
            kw_percent_on=kw_percent_on,
            context_percent_on=context_percent_on,
            weight_sparsity=weight_sparsity,
            weight_init=weight_init,
            dendrite_weight_sparsity=dendrite_weight_sparsity,
            dendrite_init=dendrite_init,
            dendritic_layer_class=dendritic_layer_class,
            output_nonlinearity=output_nonlinearity,
            preprocess_module_type=preprocess_module_type,
            preprocess_output_dim=preprocess_output_dim,
            preprocess_kw_percent_on=kw_percent_on
        )

    def forward(self, observations, actions):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.

        Returns:
            torch.Tensor: Output value
        """
        obs_only = observations[:, :-self.num_tasks]
        context_only = observations[:, -self.num_tasks:]

        if self.input_data == "obs":
            obs_portion = torch.cat([obs_only, actions], 1)
        elif self.input_data == "obs|context":
            obs_portion = torch.cat([obs_only, actions, context_only], 1)
        
        if self.context_data == "context":
            context_portion = context_only
        elif self.context_data == "obs|context":
            context_portion = torch.cat([obs_only, actions, context_only], 1)

        return super().forward(obs_portion, context_portion)
