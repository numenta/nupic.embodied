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
Base Multitask Experiment configuration.
"""

from copy import deepcopy

base = dict()

debug = deepcopy(base)
debug = dict(
    evaluation_frequency=1
)

multiseg_mt10_base = dict(
    num_segments=10,
    dendritic_layer_class="abs_max_gating",
    dim_context=10,
    num_tasks=10, 
    cpus_per_workers=7/50,
)

singleseg_mt10_base = dict(
    num_segments=1,
    dim_context=10,
    num_tasks=10,
    cpus_per_worker=7/50,
)

# Export configurations in this file
CONFIGS = dict(
    base=base,
    debug=debug,
    multiseg_mt10_base=multiseg_mt10_base,
    singleseg_mt10_base=singleseg_mt10_base,
)
