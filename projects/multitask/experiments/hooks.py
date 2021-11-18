#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Multitask Experiment configuration to test policy data collection hooks
"""

from copy import deepcopy
import torch
from collections import defaultdict

from nupic.embodied.multitask.hooks.sparse_viz import (
    AverageSegmentActivationsHook,
    HiddenActivationsPercentOnHook,
    CombinedSparseVizHook
)

from .multiseg_experiments import no_overlap_10d_abs_max_signed, dendrites_relu

class HookManagerSample:
    """
    Requires:
    - assigning a function to collect_hook_data in the recipient network
    - attaching a hook to the recipient network
    - a class method called consolidate_and_report that executes an action
    based on the data reported
    """

    def __init__(self, network):
        self.hook_data = []
        # redirect function to the network
        network.collect_hook_data = self.export_data
        # attach hook
        network.module.mean_log_std.register_forward_hook(
            self.forward_hook
        )

    def forward_hook(self, m, i, o):
        self.hook_data.append(i[0][0])

    def export_data(self):
        """Returns current data and reinitializes collection"""
        data_to_export = self.hook_data
        self.hook_data = []
        return data_to_export

    @classmethod
    def consolidate_and_report(cls, data):
        """
        Accepts a dictionary where key is the task index
        and value is a list with one entry per step take

        Class method, requires data argument

        Returns a dictionary that can be incorporated into a regular log dict
        """
        sum_inputs_per_task = defaultdict(int)
        for task_id, task_data in data.items():
            for step_data in task_data:
                sum_inputs_per_task[task_id] += torch.sum(step_data).item()
        print(sum_inputs_per_task)

        return {"sum_inputs_per_task": sum_inputs_per_task.values()}


debug = deepcopy(no_overlap_10d_abs_max_signed)
debug.update(
    evaluation_frequency=1,
    timesteps=100000,
    buffer_batch_size=32,
    num_grad_steps_scale=0.01,
)

test_hook = deepcopy(debug)
test_hook.update(
    policy_data_collection_hook=HookManagerSample,
)

test_sparse_hook = deepcopy(debug)
test_sparse_hook.update(
    policy_data_collection_hook=CombinedSparseVizHook,
)

no_overlap_10d_abs_max_signed_with_plots = deepcopy(no_overlap_10d_abs_max_signed)
no_overlap_10d_abs_max_signed_with_plots.update(
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)


no_overlap_10d_abs_max_signed_with_plots_mt50 = deepcopy(no_overlap_10d_abs_max_signed_with_plots)
no_overlap_10d_abs_max_signed_with_plots_mt50.update(
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    num_tasks=50,
    cpus_per_worker=0.14
)


dendrites_with_plots_noenvupdate = deepcopy(no_overlap_10d_abs_max_signed)
dendrites_with_plots_noenvupdate.update(
    task_update_frequency=1e12,  # fix it
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    share_train_eval_env=True,
)

dendrites_relu_with_plots = deepcopy(dendrites_relu)
dendrites_relu_with_plots.update(
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)

# Export configurations in this file
CONFIGS = dict(
    test_hook=test_hook,
    test_sparse_hook=test_sparse_hook,
    no_overlap_10d_abs_max_signed_with_plots=no_overlap_10d_abs_max_signed_with_plots,
    dendrites_with_plots_noenvupdate=dendrites_with_plots_noenvupdate,
    no_overlap_10d_abs_max_signed_with_plots_mt50=no_overlap_10d_abs_max_signed_with_plots_mt50,
    dendrites_relu_with_plots=dendrites_relu_with_plots
)