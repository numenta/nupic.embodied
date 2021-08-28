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
from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.torch.modules import KWinners, SparseWeights
import torch
from torch import nn


class CustomDendriticMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_sizes,
                 dim_context,
                 kw,
                 hidden_sizes=(32, 32),
                 num_segments=1,
                 kw_percent_on=0.05,
                 context_percent_on=1.0,
                 weight_sparsity=0.50,
                 weight_init="modified",
                 dendrite_init="modified",
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer,
                 output_nonlinearity=None,
                 preprocess_module_type=None,
                 preprocess_output_dim=128,
                 preprocess_kw_percent_on=0.1,
                 representation_module_type=None,
                 representation_module_dims=(128, 128),
                 ):
        super(CustomDendriticMLP, self).__init__()
        assert preprocess_module_type in (None, "relu", "kw")

        self.weight_sparsity = weight_sparsity

        self.representation_dim = input_dim
        # representation module: learns context(Task) independent representation of input
        self.representation_module = self._create_representation_module(
            representation_module_type,
            representation_module_dims
        )

        self.context_representation_dim = dim_context
        # preprocess module: builds a representation of context + input representation (for input to dendrite segments)
        self.preprocess_module = self._create_preprocess_module(
            preprocess_module_type,
            preprocess_output_dim,
            preprocess_kw_percent_on
        )

        self.dendritic_module = DendriticMLP(
            input_size=self.representation_dim,
            output_size=output_sizes,
            hidden_sizes=hidden_sizes,
            num_segments=num_segments,
            dim_context=self.context_representation_dim,
            kw=kw,
            kw_percent_on=kw_percent_on,
            context_percent_on=context_percent_on,
            weight_sparsity=weight_sparsity,
            weight_init=weight_init,
            dendrite_init=dendrite_init,
            dendritic_layer_class=dendritic_layer_class,
            output_nonlinearity=output_nonlinearity,
        )

    def forward(self, x, context):
        if self.representation_module is not None:
            x = self.representation_module(x)

        if self.context_representation_dim is not None:
            context = self.preprocess_module(torch.cat([x, context], dim=-1))

        return self.dendritic_module(x, context)

    def _create_representation_module(self, module_type, dims):
        if module_type is None:
            return None
        representation_module = nn.Sequential()

        inp_dim = self.input_size
        for i in range(len(dims)):
            output_dim = dims[i]
            layer = SparseWeights(
                torch.nn.Linear(inp_dim,
                                output_dim,
                                bias=True),
                sparsity=self.weight_sparsity,
                allow_extremes=True
            )
            # network input is dense (no sparsity constraints)
            DendriticMLP._init_sparse_weights(layer, 0.0)

            if module_type == "relu":
                nonlinearity = nn.ReLU()
            else:
                raise NotImplementedError
            representation_module.add_module("linear_layer_{}".format(i), layer)
            representation_module.add_module("nonlinearity_{}".format(i), nonlinearity)
            inp_dim = output_dim

        self.representation_dim = inp_dim
        return representation_module

    def _create_preprocess_module(self, module_type, preprocess_output_dim, kw_percent_on):
        # TODO: kw_percent_on should be optional given relu can be a module type

        if module_type is None:
            return None

        preprocess_module = nn.Sequential()
        linear_layer = SparseWeights(
            # if m10 context_representation_dim=10, context_representation_dim=50
            torch.nn.Linear(self.context_representation_dim + self.representation_dim,
                            preprocess_output_dim,
                            bias=True),
            sparsity=self.weight_sparsity,
            allow_extremes=True
        )
        DendriticMLP._init_sparse_weights(linear_layer, 0.0)

        if module_type == "relu":
            nonlinearity = nn.ReLU()
        else:
            nonlinearity = KWinners(
                n=preprocess_output_dim,
                percent_on=kw_percent_on,
                k_inference_factor=1.0,
                boost_strength=0.0,
                boost_strength_factor=0.0
            )
        preprocess_module.add_module("linear_layer", linear_layer)
        preprocess_module.add_module("nonlinearity", nonlinearity)

        self.context_representation_dim = preprocess_output_dim
        return preprocess_module
