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


import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm, colors
from matplotlib import pyplot as plt

from nupic.research.frameworks.dendrites import DendriticLayerBase

from .base import HookManagerBase


class PolicyVisualizationsHook(HookManagerBase):

    def init_data_collection(self):
        self.targets = []
        self.activations = []

    def target_hook_fn(self, module, input, output):
        # Input is (batch size x num_tasks) matrix
        input = input[0]
        # Verify preprocess only takes in one-hot encoded vector
        # assert (input.count_nonzero(dim=1) == 1).all().item()
        # Get one-hot encoded task id from each row
        targets = (input == 1).nonzero(as_tuple=True)[1].squeeze()

        self.targets.append(targets)

    def activation_hook_fn(self, module, input, output):
        # output is (batch size x num_units x num_segments) matrix
        self.activations.append(output)

    def export_data(self):
        """Returns current data and reinitializes collection"""
        targets = torch.cat(self.targets, dim=0).to(torch.int)
        activations = torch.cat(self.activations, dim=0)
        self.init_data_collection()
        return targets, activations

    @classmethod
    def consolidate_and_report(cls, data):
        """
        Accepts a dictionary where key is the task index
        and value is a list with one entry per step take

        Class method, requires data argument
        """
        for task_id, task_data in data.items():
            for (targets, activations) in task_data:
                print(f"Visualizing data for task {task_id}")
                cls.get_visualization(targets, activations)

    @classmethod
    def get_visualization(cls, targets, activations):
        raise NotImplementedError


class AverageSegmentActivationsHook(PolicyVisualizationsHook):

    @classmethod
    def get_visualization(cls, targets, activations, unit_to_plot=0):
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
            num_segments = activations.size(2)
            num_tasks = targets.max().item() + 1
            activations = activations[:, unit_to_plot, :]

            avg_activations = torch.zeros((num_segments, 0))

            for t in range(num_tasks):
                inds_t = torch.nonzero((targets == t).float()).squeeze()

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
            plt.savefig("/Users/lsouza/nta/results/test_figures")

            return {"average_segment_activations": figure}

    def attach(self, network):
        """
        TODO: if layer is always one, remove lists, asserts and list handling methods
        """
        preprocess_layers = []
        dendrite_layers = []

        for name, layer in network.named_modules():
            if isinstance(layer, nn.Sequential) and "preprocess" in name:
                preprocess_layers.append(layer)
            elif isinstance(layer, DendriticLayerBase): # TODO: not found in network
                dendrite_layers.append(layer.segments)

        # TODO: asserts failing, finding 0 dendrite layers
        # assert len(preprocess_layers) == 1, f"Found {len(preprocess_layers)} layers"
        # assert len(dendrite_layers) == 1, f"Found {len(dendrite_layers)} layers"

        preprocess_layers[0].register_forward_hook(self.target_hook_fn)
        dendrite_layers[0].register_forward_hook(self.activation_hook_fn)

class HiddenActivationsPercentOnHook(PolicyVisualizationsHook):

    @classmethod
    def get_visualization(cls, targets, activations, num_units_to_plot=64):
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
            device = activations.device

            num_tasks = targets.max().item() + 1
            _, num_units = activations.size()

            #habu = hidden activations by unit
            habu = torch.zeros((0, num_units))
            habu = habu.to(device)

            for t in range(num_tasks):
                inds_t = torch.nonzero((targets == t).float(), as_tuple=True)

                habu_t = activations[inds_t]

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
            plt.savefig("/Users/lsouza/nta/results/test_figures")

            return {"hidden_activations_percent_on": figure}

    def attach(self, network):
        """
        TODO: if layer is always one, remove lists, asserts and list handling methods
        """
        preprocess_layers = []
        dendrite_layers = []

        for name, layer in network.named_modules():
            if isinstance(layer, nn.Sequential) and "preprocess" in name:
                preprocess_layers.append(layer)
            elif isinstance(layer, nn.Sequential) and "dendrite" in name:
                dendrite_layers.append(layer)

        # TODO: asserts failing, finding 2 preprocess layers
        # assert len(preprocess_layers) == 1, f"Found {len(preprocess_layers)} layers"
        # assert len(dendrite_layers) == 1, f"Found {len(dendrite_layers)} layers"

        preprocess_layers[0].register_forward_hook(self.target_hook_fn)
        dendrite_layers[0].register_forward_hook(self.activation_hook_fn)


class CombinedSparseVizHook():

    hooks_classes = [
        # AverageSegmentActivationsHook,
        HiddenActivationsPercentOnHook
    ]

    def __init__(self, network):
        self.hooks = [hook(network) for hook in self.hooks_classes]

    def export_data(self):
        # Output of each hook is a tuple target, activations
        return {hook.__class__: hook.export_data() for hook in self.hooks}

    @classmethod
    def consolidate_and_report(cls, data):
        """
        Accepts a dictionary where key is the task index
        and value is a list with one entry per step take

        Class method, requires data argument
        """
        # Loop through tasks
        visualizations = {}
        for _, task_data in data.items():
            # Loop through steps
            for step_data in task_data:
                # Loop through hooks - new in combined
                for hook in cls.hooks_classes:
                    targets, activations = step_data[hook.__class__]
                    visualizations.update(hook.get_visualization(targets, activations))

        return visualizations
