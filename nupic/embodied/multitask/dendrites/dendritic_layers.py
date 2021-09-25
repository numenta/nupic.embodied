# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
A simple implementation of dendrite weights. This combines the output from a (sparse)
linear layer with the output from a set of dendritic segments.
"""
import abc

import torch

from nupic.research.frameworks.dendrites import DendriteSegments
from nupic.torch.modules.sparse_weights import SparseWeights


class DendriteBase(SparseWeights, metaclass=abc.ABCMeta):
    """
    Base class for all Dendritic Layer modules.

    This combines a DendriteSegments module with a SparseLinear module.
    The output from the dendrite segments (shape of num_units x num_segments)
    is applied to the output of of the linear weights (shape of num_units).
    Thus, each linear output unit gets modulated by a set of dendritic segments.
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        :param module: linear module from in-units to out-units
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.dim_context = dim_context

        self.segments = None
        super().__init__(
            module,
            sparsity=module_sparsity,
            allow_extremes=True
        )

        self.segments = DendriteSegments(
            num_units=module.weight.shape[0],
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations, training_mode):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context, training_mode):
        """Compute of linear layer and apply output of dendrite segments."""
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations, training_mode)

    @property
    def segment_weights(self):
        return self.segments.weights


class AbsoluteMaxGatingDendriticLayer(DendriteBase):
    """
    This layer is similar to `GatingDendriticLayer`, but selects dendrite activations
    based on absolute max activation values instead of just max activation values. For
    example, if choosing between activations -7.4, and 6.5 for a particular unit, -7.4
    will be chosen, and its sign will be kept.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval_optimized = True

    def apply_dendrites(self, y, dendrite_activations, training_mode):
        """Apply dendrites as a gating mechanism."""

        if training_mode:
            return (y * torch.sigmoid(dendrite_activations.abs().amax(dim=2)))
        else:
            return (y * torch.sigmoid(dendrite_activations.square().amax(dim=2)))


class FFLayer(SparseWeights):
    """
    Class for a layer of units with no dendritic segments per unit. This is identical
    to a normal feed-forward layer, but useful for debugging to ensure we use the same
    code paths and that everything else is identical.
    """

    def __init__(self, module, module_sparsity):
        """
        :param module: linear module from in-units to out-units
        :param module_sparsity: sparsity applied over linear module
        """
        super().__init__(module,
                         sparsity=module_sparsity,
                         allow_extremes=True)

        self.rezero_weights()

    def forward(self, x, context):
        return super().forward(x)