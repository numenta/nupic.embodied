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
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nupic.research.frameworks.dendrites import DendriticLayerBase

class PolicyVisualizationHook(metaclass=abc.ABCMeta):
    def __init__(self, net):
        self.net = net
    
    @abc.abstractmethod
    def attach(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_visualization(self):
        raise NotImplementedError


class AverageSegmentActivationsHook(PolicyVisualizationHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.targets = torch.Tensor([])
        self.activations = torch.Tensor([])

    def target_hook_fn(self, module, input, output):
        # input is (batch size x num_tasks) matrix
        input = input[0]

        # verify preprocess only takes in one-hot encoded vector
        assert (input.count_nonzero(dim=1) == 1).all().item()
        
        # get one-hot encoded task id from each row
        targets = (input == 1).nonzero(as_tuple=True)[1].squeeze()

        self.targets = torch.cat([self.targets, targets], dim=0).to(torch.int)

    def activation_hook_fn(self, module, input, output):
        # output is (batch size x num_units x num_segments) matrix

        self.activations = torch.cat([self.activations, output], dim=0)

    def get_visualization(self, unit_to_plot=0):
        """
        Returns a heatmap of dendrite activations for a single unit, plotted using
        matplotlib.
        :param dendrite_activations: 3D torch tensor with shape (batch_size, num_units,
                                    num_segments) in which entry b, i, j gives the
                                    activation of the ith unit's jth dendrite segment for
                                    example b
        :param unit_to_plot: index of the unit for which to plot dendrite activations;
                            plots activations of unit 0 by default
        """

        with torch.no_grad():
            num_segments = self.activations.size(2)
            num_tasks = self.targets.max().item() + 1
            activations = self.activations[:, unit_to_plot, :]
            
            avg_activations = torch.zeros((num_segments, 0))

            for t in range(num_tasks):
                inds_t = torch.nonzero((self.targets == t).float()).squeeze()

                activations_t = activations[inds_t, :].mean(dim=0).detach().cpu().unsqueeze(dim=1)

                avg_activations = torch.cat((avg_activations, activations_t), dim=1)

            vmax = avg_activations.abs().max().item()
            vmin = -1.0 * vmax

            ax = plt.gca()

            ax.imshow(avg_activations.detach().cpu().numpy(), cmap="coolwarm_r", vmin=vmin, vmax=vmax)
            plt.colorbar(
                cm.ScalarMappable(norm=colors.Normalize(-1, 1), cmap="coolwarm_r"), ax=ax, location="left",
                shrink=0.6, drawedges=False, ticks=[-1.0, 0.0, 1.0]
            )

            ax.set_xlabel("Task")
            ax.set_ylabel("Segment")
            ax.set_xticks(range(num_tasks))
            ax.set_yticks(range(num_segments))

            plt.tight_layout()
            figure = plt.gcf()

            return "average_segment_activations", figure

        
    def attach(self):
        preprocess_layers = []
        dendrite_layers = []

        for name, layer in self.net.named_modules():
            if isinstance(layer, nn.Sequential) and "preprocess" in name:
                preprocess_layers.append(layer)
            elif isinstance(layer, DendriticLayerBase):
                dendrite_layers.append(layer.segments)
            

        # only 1 preprocess and 1 dendrite layer
        assert len(preprocess_layers) == 1
        assert len(dendrite_layers) == 1

        preprocess_layers[0].register_forward_hook(self.target_hook_fn)
        dendrite_layers[0].register_forward_hook(self.activation_hook_fn)


class HiddenActivationsPercentOnHook(PolicyVisualizationHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.targets = torch.Tensor([])
        self.activations = torch.Tensor([])

    def target_hook_fn(self, module, input, output):
        # input is (batch size x num_tasks) matrix
        input = input[0]

        # verify preprocess only takes in one-hot encoded vector
        assert (input.count_nonzero(dim=1) == 1).all().item()
        
        # get one-hot encoded task id from each row
        targets = (input == 1).nonzero(as_tuple=True)[1].squeeze()

        self.targets = torch.cat([self.targets, targets], dim=0).to(torch.int)

    def activation_hook_fn(self, module, input, output):
        # input/output is (batch size x hidden dim) matrix

        self.activations = torch.cat([self.activations, output], dim=0)

    def get_visualization(self, num_units_to_plot=64):
        """
        Returns a heatmap with shape (num_categories, num_units) where cell c, i gives the
        mean value of hidden activations for unit i over all given examples from category
        c.
        :param activations: 2D torch tensor with shape (batch_size, num_units) where entry
                            b, i gives the activation of unit i for example b
        :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                        target label for example b
        :param num_units_to_plot: an integer which gives how many columns to show, for ease
                                of visualization; only the first num_units_to_plot units
                                are shown
        """

        with torch.no_grad():
            device = self.activations.device

            num_tasks = self.targets.max().item() + 1
            _, num_units = self.activations.size()

            #habu = hidden activations by unit
            habu = torch.zeros((0, num_units))
            habu = habu.to(device)

            for t in range(num_tasks):
                inds_t = torch.nonzero((self.targets == t).float(), as_tuple=True)

                habu_t = self.activations[inds_t]

                habu_t = habu_t.mean(dim=0).unsqueeze(dim=0)
                habu = torch.cat((habu, habu_t))

            habu = habu[:, :num_units_to_plot].detach().cpu().numpy()

            max_val = np.abs(habu).max()

            ax = plt.gca()

            ax.imshow(habu, cmap="Greens", vmin=0, vmax=max_val)
            plt.colorbar(
                cm.ScalarMappable(cmap="Greens"), ax=ax, location="top",
                shrink=0.5, ticks=[0.0, 0.5, 1.0], drawedges=False
            )

            ax.set_aspect(2.5)
            ax.set_xlabel("Hidden unit")
            ax.set_ylabel("Task")
            ax.get_yaxis().set_ticks(range(num_tasks))
            
            plt.tight_layout()
            figure = plt.gcf()

            return "hidden_activations_percent_on", figure
        
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