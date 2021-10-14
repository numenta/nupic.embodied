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
import torch.nn as nn
import abc
from collections import defaultdict

from nupic.research.frameworks.dendrites.metrics import plot_hidden_activations_by_unit


class HookManager(metaclass=abc.ABCMeta):
    def __init__(self, net):
        self.net = net
    
    @abc.abstractmethod
    def attach(self):
        raise NotImplementedError


class PolicyHook(HookManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.targets = torch.Tensor([])
        self.activations = torch.Tensor([])

    def target_hook_fn(self, module, input, output):
        # input is (batch size x num_tasks) matrix

        # verify preprocess only takes in one-hot encoded vector
        assert (input.count_nonzero(dim=1) == 1).all().item()
        
        # get one-hot encoded task id from each row
        targets = (input == 1).nonzero(as_tuple=True)[0].squeeze()

        self.targets = torch.cat([self.targets, targets], dim=0)

    def activation_hook_fn(self, module, input, output):
        # input/output is (batch size x hidden dim) matrix

        self.activations = torch.cat([self.activations, output], dim=0)

    def get_visualization(self):
        return plot_hidden_activations_by_unit(self.activations, self.targets)
        
    def attach(self):
        preprocess_layers = []
        dendrite_layers = []

        for name, layer in self.net.named_modules():
            if isinstance(layer, nn.Sequential) and "preprocess" in name:
                preprocess_layers.append(layer)
            elif isinstance(layer, nn.Sequential) and "dendrite" in name:
                dendrite_layers.append(layer)

        # only 1 preprocess and 1 dendrite layer
        assert len(preprocess_layers) == 1
        assert len(dendrite_layers) == 1

        preprocess_layers[0].register_forward_hook(self.target_hook_fn)
        dendrite_layers[0].register_forward_hook(self.activation_hook_fn)

        