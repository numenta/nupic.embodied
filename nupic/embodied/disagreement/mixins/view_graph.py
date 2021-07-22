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

from torchviz import make_dot
from matplotlib import pyplot as plt


class ViewGraph(object):
    """
    Mixin to PPOOptimizer (agent) to allow visualization of disagreement
    computational graph
    """

    def calculate_disagreement(self, acs, features, next_features):
        """If next features is defined, return prediction error.
        Otherwise returns predictions i.e. dynamics model last layer output
        """

        disagreement = super().calculate_disagreement(acs, features, next_features)

        model = self.dynamics_list[0].dynamics_net

        print("********************")
        print("Making and saving plot")
        dot = make_dot(
            disagreement,
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True
        )
        dot.format = "png"
        dot.render()
        # plt.savefig("disagreement_comp_graph.png")

        return disagreement
