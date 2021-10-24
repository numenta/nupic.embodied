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

from .base import debug
from copy import deepcopy
import torch
from collections import defaultdict


class HookManagerSample:
    """
    Requires:
    - assigning a function to collect_log_data in the recipient network
    - attaching a hook to the recipient network
    - a class method called consolidate_and_report that executes an action
    based on the data reported
    """

    def __init__(self, network):
        self.hook_data = []
        # redirect function to the network
        network.collect_log_data = self.export_data
        # attach hook
        network._module._shared_mean_log_std_network.register_forward_hook(
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
        """
        sum_inputs_per_task = defaultdict(int)
        for task_id, task_data in data.items():
            for step_data in task_data:
                sum_inputs_per_task[task_id] += torch.sum(step_data).item()
        print(sum_inputs_per_task)


test_hook = deepcopy(debug)
test_hook.update(
    policy_data_collection_hook=HookManagerSample,
    evaluation_frequency=1,
)

# Export configurations in this file
CONFIGS = dict(
    test_hook=test_hook
)