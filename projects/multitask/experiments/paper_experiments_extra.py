# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

from copy import deepcopy
from collections import defaultdict
from nupic.embodied.multitask.dendrites import dendritic_layers
import torch

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

'''
GENERAL NOTE: ALL RUNS USE A RANDOM ENVIRONMENT SEED (SOME RANDOMLY FIXED GOAL)
'''

dendrite_both_layers_modulated = dict(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    layers_modulated=(0, 1),
    kw_percent_on=0.25, 
    weight_sparsity=(1-0.9),
    fp16=True,
    num_tasks=10,
    num_segments=10,
    net_type="Dendrite_MLP",
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    project_name="multitask-journal-extra",
    wandb_group="Paper Figures Extra",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    cpus_per_worker=0.5,
    gpus_per_worker=0,
)

dendrite_one_layer = dict(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(4000,),
    layers_modulated=(0,),
    kw_percent_on=0.25, 
    weight_sparsity=(1-0.9),
    fp16=True,
    num_tasks=10,
    num_segments=10,
    net_type="Dendrite_MLP",
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    project_name="multitask-journal-extra",
    wandb_group="Paper Figures Extra",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    cpus_per_worker=0.5,
    gpus_per_worker=0,
)

dendrite_three_layer = dict(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2000, 2000, 2000),
    layers_modulated=(2,),
    kw_percent_on=0.25, 
    weight_sparsity=(1-0.9),
    fp16=True,
    num_tasks=10,
    num_segments=10,
    net_type="Dendrite_MLP",
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    project_name="multitask-journal-extra",
    wandb_group="Paper Figures Extra",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    cpus_per_worker=0.5,
    gpus_per_worker=0,
)

dendrite_20seg = dict(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    layers_modulated=(1,),
    kw_percent_on=0.25, 
    weight_sparsity=(1-0.9),
    fp16=True,
    num_tasks=10,
    num_segments=20,
    net_type="Dendrite_MLP",
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    project_name="multitask-journal-extra",
    wandb_group="Paper Figures Extra",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    cpus_per_worker=0.5,
    gpus_per_worker=0,
)


CONFIGS = dict(
    dendrite_both_layers_modulated=dendrite_both_layers_modulated,
    dendrite_one_layer=dendrite_one_layer,
    dendrite_three_layer=dendrite_three_layer,
    dendrite_20seg=dendrite_20seg
)