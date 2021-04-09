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
import torch
from torch import nn

from functools import partial


class DynamicsLeakyRelu(nn.LeakyReLU):
    """Default activation function for Dynamic Networks"""

    def forward(self, x, ac):
        return super().forward(x), ac


class DynamicsLinear(nn.Linear):
    """Default linear layer for Dynamic Networks"""

    def forward(self, x, ac):
        x = torch.cat((x, ac), dim=-1)
        return super().forward(x), ac


class DynamicsSequential(nn.Sequential):
    """Sequential for Dynamic Networks that accepts action"""

    def forward(self, x, ac):
        for module in self:
            x, _ = module(x, ac)
        return x, ac


class DynamicsBlock(DynamicsSequential):
    """
    Default hidden block for Dynamic Networks
    Two linear layers each, the first layer uses a non-linear activation function.
    """

    def __init__(self, hidden_dim, ac_dim, activation_fn):
        super().__init__(
            DynamicsLinear(hidden_dim + ac_dim, hidden_dim),
            activation_fn(),
            DynamicsLinear(hidden_dim + ac_dim, hidden_dim),
        )

    def forward(self, x, ac):
        out, _ = super().forward(x, ac)
        return out + x, ac


class DynamicsNet(nn.Module):
    """Residual network to get the dynamics loss using the features from the auxiliary
    task model.

    :param nblocks: Number of residual blocks in the dynamics network.
    :param feature_dim: Number of features from the feature network.
    :param ac_dim: Action dimensionality.
    :param out_feature_dim: Number of features from the feature network for the next
                    state (usually same).
    :param hidden_dim: Number of neurons in the hidden layers.
    :param activation_fn: Activation function factory.
    """

    leaky_relu = partial(DynamicsLeakyRelu, negative_slope=0.2)

    def __init__(
        self,
        nblocks,
        feature_dim,
        ac_dim,
        out_feature_dim,
        hidden_dim,
        activation_fn=leaky_relu,
    ):
        super().__init__()

        # First layer of the model takes state features + actions as input and outputs
        # hidden_dim activations
        self.input = DynamicsSequential(
            DynamicsLinear(feature_dim + ac_dim, hidden_dim),
            activation_fn(),
        )

        # n residual blocks
        self.hidden = DynamicsSequential(
            *[DynamicsBlock(hidden_dim, ac_dim, activation_fn) for _ in range(nblocks)]
        )

        self.output = DynamicsLinear(hidden_dim + ac_dim, out_feature_dim)

        self.init_weight()

    def init_weight(self):
        """Initialize the weights with xavier (glorot) uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x, ac):
        """Get output of a forward pass through the dynamics network.

        :param features: Features from the auxiliary network corresponding with the
                         current state
        :param ac: Current actions.
        :return: Features of residual dynamics model from the current states & actions
        """
        x, _ = self.input(x, ac)
        x, _ = self.hidden(x, ac)
        x, _ = self.output(x, ac)
        return x
