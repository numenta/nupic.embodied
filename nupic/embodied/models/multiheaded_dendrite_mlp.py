import torch.nn as nn
from nupic.torch.modules import KWinners
from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    DendriticAbsoluteMaxGate1d,
    DendriticGate1d
)
from garage.torch import NonLinearity

class MultiHeadedDendriticMLP(nn.Module):
    """
    A dendritic network which is similar to a MLP with a two hidden layers, except that
    activations are modified by dendrites. The context input to the network is used as
    input to the dendritic weights.
                    _____
                   |_____|    # Classifier layer, no dendrite input
                      ^
                      |
                  _________
    context -->  |_________|  # Second linear layer with dendrites
                      ^
                      |
                  _________
    context -->  |_________|  # First linear layer with dendrites
                      ^
                      |
                    input
    """

    def __init__(self, input_size, num_heads, output_dims, dim_context, hidden_sizes=(32, 32), num_segments=(5, 5),
                 sparsity=0.5, kw=False, relu=False, output_nonlinearities=None,
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):

        assert dendritic_layer_class in {AbsoluteMaxGatingDendriticLayer, DendriticAbsoluteMaxGate1d, DendriticGate1d}

        # The nonlinearity can either be k-Winners or ReLU, but not both
        assert not (kw and relu)
        assert num_heads == len(output_dims)

        super().__init__()

        self.num_heads = num_heads
        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_dims = output_dims
        self.dim_context = dim_context
        self.kw = kw
        self.relu = relu

        self._layers = []
        self._activations = []
        prev_dim = input_size
        for i in range(len(hidden_sizes)):
            curr_dend = dendritic_layer_class(
                module=nn.Linear(prev_dim, hidden_sizes[i]),
                num_segments=num_segments[i],
                dim_context=dim_context,
                module_sparsity=sparsity,
                dendrite_sparsity=sparsity
            )
            if kw:
                curr_activation = KWinners(n=hidden_sizes[i], percent_on=0.05, k_inference_factor=1.0,
                                           boost_strength=0.0, boost_strength_factor=0.0)
            else:
                curr_activation = nn.ReLU()

            self._layers.append(curr_dend)
            self._activations.append(curr_activation)
            prev_dim = hidden_sizes[i]

        # Final multiheaded layer
        self._output_layers = nn.ModuleList()
        for i in range(self.num_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(prev_dim, output_dims[i])
            output_layer.add_module('linear', linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module('non_linearity',
                                        NonLinearity(output_nonlinearities[i]))
            self._output_layers.append(output_layer)

    def forward(self, x, context):
        for layer, activation in zip(self._layers, self._activations):
            x = activation(layer(x, context))

        return [output_layer(x) for output_layer in self._output_layers]