from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
)
from nupic.embodied.models import MultiHeadedDendriticMLP

class DendriticMLP(MultiHeadedDendriticMLP):
    """
    A dendritic network which is similar to a MLP with a two hidden layers, except that
    activations are modified by dendrites. The context input to the network is used as
    input to the dendritic weights.
    """

    def __init__(self, input_size, output_dim, dim_context, hidden_sizes=(32, 32), num_segments=(5, 5),
                 sparsity=0.5, kw=False, relu=False, output_nonlinearities=None,
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):

        super(DendriticMLP, self).__init__(
            input_size=input_size,
            num_heads=1,
            output_dims=(output_dim),
            dim_context=dim_context,
            hidden_sizes=hidden_sizes,
            num_segments=num_segments,
            sparsity=sparsity,
            kw=kw,
            relu=relu,
            output_nonlinearities=output_nonlinearities,
            dendritic_layer_class=dendritic_layer_class
        )
        self._output_dim = output_dim

    def forward(self, x, context):
        return super().forward(x, context)[0]

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim
