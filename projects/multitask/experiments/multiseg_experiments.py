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

from .base import multiseg_mt10_base

from copy import deepcopy

seg10 = dict(
    num_segments=10,
)
seg10.update(multiseg_mt10_base)

seg5 = dict(
    num_segments=5,
)
seg5.update(multiseg_mt10_base)


'''
Round 1 of experiments
'''

baseline_5d = deepcopy(seg5)
baseline_5d.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(350, 350),
    kw_percent_on=0.25,
    weight_sparsity=0.5,
    fp16=True,
    preprocess_output_dim=32,
)

baseline_5d_bigger = deepcopy(seg5)
baseline_5d_bigger.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1000, 1000),
    kw_percent_on=0.25,
    weight_sparsity=0.25,
    fp16=True,
    preprocess_output_dim=32,
)

baseline_10d = deepcopy(seg10)
baseline_10d.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(232, 232),
    kw_percent_on=0.25,
    weight_sparsity=0.5,
    fp16=True,
    preprocess_output_dim=32,
)


'''
Round 2 of experiments
'''

no_overlap_10d = deepcopy(seg10)
no_overlap_10d.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_unsigned",
)

overlap_input_10d = deepcopy(seg10)
overlap_input_10d.update(
    input_data="obs|context",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_unsigned",
)

overlap_context_10d = deepcopy(seg10)
overlap_context_10d.update(
    input_data="obs",
    context_data="obs|context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1600, 1600),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=20,
    dendritic_layer_class="abs_max_gating_unsigned",
)

overlap_both_10d = deepcopy(seg10)
overlap_both_10d.update(
    input_data="obs|context",
    context_data="obs|context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1600, 1600),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=20,
    dendritic_layer_class="abs_max_gating_unsigned",
)

'''
Sanity checking Round 2 results
'''

no_overlap_10d_preprocess_relu = deepcopy(seg10)
no_overlap_10d_preprocess_relu.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="max_gating",
)

no_overlap_10d_preprocess_kw = deepcopy(seg10)
no_overlap_10d_preprocess_kw.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_module_type="kw",
    preprocess_output_dim=10,
    dendritic_layer_class="max_gating",
)

no_overlap_10d_preprocess_none = deepcopy(seg10)
no_overlap_10d_preprocess_none.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_module_type=None,
    preprocess_output_dim=10,
    dendritic_layer_class="max_gating",
)


no_overlap_10d_abs_max_unsigned = deepcopy(seg10)
no_overlap_10d_abs_max_unsigned.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_unsigned",
)

no_overlap_15d_abs_max_unsigned = deepcopy(seg10)
no_overlap_15d_abs_max_unsigned.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    num_segments=15,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_unsigned",
)

no_overlap_10d_abs_max_signed = deepcopy(seg10)
no_overlap_10d_abs_max_signed.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.10,
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Paper Figures",
    evaluation_frequency=5,
)


CONFIGS = dict(
    baseline_5d=baseline_5d,
    baseline_10d=baseline_10d,
    baseline_5d_bigger=baseline_5d_bigger,
    no_overlap_10d=no_overlap_10d,
    overlap_input_10d=overlap_input_10d,
    overlap_context_10d=overlap_context_10d,
    overlap_both_10d=overlap_both_10d,
    no_overlap_10d_preprocess_relu=no_overlap_10d_preprocess_relu,
    no_overlap_10d_preprocess_kw=no_overlap_10d_preprocess_kw,
    no_overlap_10d_preprocess_none=no_overlap_10d_preprocess_none,
    no_overlap_10d_abs_max_unsigned=no_overlap_10d_abs_max_unsigned,
    no_overlap_15d_abs_max_unsigned=no_overlap_15d_abs_max_unsigned,
    no_overlap_10d_abs_max_signed=no_overlap_10d_abs_max_signed,
)