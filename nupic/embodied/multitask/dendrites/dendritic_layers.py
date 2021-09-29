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

from nupic.research.frameworks.dendrites import DendriticLayerBase
from nupic.torch.modules.sparse_weights import SparseWeights


class AbsoluteMaxGatingUnsignedDendriticLayer(DendriticLayerBase):
    """
    This layer performs abs max gating (unsigned). 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""

        return (y * torch.sigmoid(dendrite_activations.abs().amax(dim=2)))


class MaxGatingDendriticLayer(DendriticLayerBase):
    """
    This layer performs max gating. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""

        return (y * torch.sigmoid(dendrite_activations.amax(dim=2)))
        

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