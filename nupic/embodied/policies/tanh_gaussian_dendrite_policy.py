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
"""TanhGaussianDendritePolicy."""
import numpy as np

from garage.torch.distributions import TanhNormal
from garage.torch.policies.stochastic_policy import StochasticPolicy
from nupic.embodied.modules.gaussian_dendrite_module import (
    GaussianDendriteTwoHeadedModule,
)
from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer

MIN_STD = np.exp(-20.)
MAX_STD = np.exp(2.)


class TanhGaussianDendriticPolicy(StochasticPolicy):
    """Multiheaded Dendritic MLP whose outputs are fed into a TanhNormal distribution.

    A policy that contains a Dendritic MLP to make prediction based on a gaussian
    distribution with a tanh transformation.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the Gaussian MLP. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 dim_context,
                 hidden_sizes=(32, 32),
                 num_segments=(5, 5),
                 sparsity=0.5,
                 kw=False,
                 relu=False,
                 mean_nonlinearity=None,
                 std_nonlinearity=None,
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer,
                 init_std=1.0,
                 min_std=MIN_STD,
                 max_std=MAX_STD,
                 std_parameterization="exp",
                 ):
        super().__init__(env_spec, name="TanhGaussianPolicy")

        # this is usually a 1-hot vector denoting the task
        self._dim_context = dim_context
        # obs space in env spec has shape obs_dim + dim_context
        self._obs_dim = env_spec.observation_space.flat_dim - self._dim_context
        self._action_dim = env_spec.action_space.flat_dim

        self._module = GaussianDendriteTwoHeadedModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            dim_context=self._dim_context,
            hidden_sizes=hidden_sizes,
            num_segments=num_segments,
            sparsity=sparsity,
            kw=kw,
            relu=relu,
            mean_nonlinearity=mean_nonlinearity,
            std_nonlinearity=std_nonlinearity,
            dendritic_layer_class=dendritic_layer_class,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            normal_distribution_cls=TanhNormal
        )

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        # separate the env observation into true observation and context
        obs_portion = observations[:, :self._obs_dim]
        context_portion = observations[:, self._obs_dim:]
        dist = self._module(obs_portion, context_portion)
        ret_mean = dist.mean.cpu()
        ret_log_std = (dist.variance.sqrt()).log().cpu()
        return dist, dict(mean=ret_mean, log_std=ret_log_std)
