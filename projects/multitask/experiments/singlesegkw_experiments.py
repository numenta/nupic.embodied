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

from .base import singleseg_mt10_base 

from copy import deepcopy

singlesegkw_base = dict()
singlesegkw_base.update(singleseg_mt10_base)

# single segment K-winners experiment 
singlesegkw1 = deepcopy(singlesegkw_base)
singlesegkw1.update(
    policy_lr=3.85e-4,
    qf_lr=3.85e-4,
    hidden_sizes=(2048, 2048, 2048),
    kw_percent_on=0.17,
    fp16=True
)

singlesegkw2 = deepcopy(singlesegkw_base)
singlesegkw2.update(
    policy_lr=3.85e-4,
    qf_lr=3.85e-4,
    hidden_sizes=(2048, 2048, 2048),
    kw_percent_on=0.17,
    fp16=False
)

CONFIGS = dict(
    singlesegkw1=singlesegkw1,
    singlesegkw2=singlesegkw2,
)