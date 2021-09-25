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

from .base import mlp_mt10_base

from copy import deepcopy

'''
Round 1 of experiments
'''

metaworld_base = deepcopy(mlp_mt10_base)
metaworld_base.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(400, 400),
    kw_percent_on=None,
    fp16=True,
)

gradient_surgery_base = deepcopy(mlp_mt10_base)
gradient_surgery_base.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(160, 160, 160, 160, 160, 160),
    kw_percent_on=None,
    fp16=True,
)

baseline_similar_metaworld = deepcopy(mlp_mt10_base)
baseline_similar_metaworld.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(580, 580),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=0.5,
)

baseline_similarv2_metaworld = deepcopy(mlp_mt10_base)
baseline_similarv2_metaworld.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1000, 1000),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=0.5,
)


'''
Round 2 of experiments
'''

new_metaworld_baseline = deepcopy(mlp_mt10_base)
new_metaworld_baseline.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(750, 750),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
)


CONFIGS = dict(
    metaworld_base=metaworld_base,
    gradient_surgery_base=gradient_surgery_base,
    baseline_similar_metaworld=baseline_similar_metaworld,
    baseline_similarv2_metaworld=baseline_similarv2_metaworld,
    new_metaworld_baseline=new_metaworld_baseline,
)