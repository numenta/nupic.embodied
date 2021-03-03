import torch.nn as nn
from nupic.research.frameworks.pytorch.models.le_sparse_net import add_sparse_linear_layer
from garage.torch import NonLinearity

class MultiHeadedSparseMLP(nn.Module):
    def __init__(self, input_size,
                 output_dims,
                 output_nonlinearities=None,
                 hidden_sizes=(32, 32),
                 linear_activity_percent_on=(0.1,0.1),
                 linear_weight_percent_on=(0.4,0.4),
                 boost_strength=1.67,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 consolidated_sparse_weights=False,
                 ):
        super(MultiHeadedSparseMLP, self).__init__()

        self.n_heads = len(output_dims)
        self._hidden_base = nn.ModuleList()
        self._hidden_base.add_module("flatten", nn.Flatten())
        # Add Sparse Linear layers
        for i in range(len(hidden_sizes)):
            add_sparse_linear_layer(
                network=self._hidden_base,
                suffix=i + 1,
                input_size=input_size,
                linear_n=hidden_sizes[i],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                weight_sparsity=linear_weight_percent_on[i],
                percent_on=linear_activity_percent_on[i],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
                consolidated_sparse_weights=consolidated_sparse_weights,
            )
            input_size = hidden_sizes[i]

        self._output_layers = nn.ModuleList()
        self._output_layers = nn.ModuleList()

        for i in range(self.n_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(input_size, output_dims[i])
            output_layer.add_module('linear', linear_layer)

            if output_nonlinearities:
                output_layer.add_module('non_linearity',
                                        NonLinearity(output_nonlinearities[i]))
            self._output_layers.append(output_layer)

    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]
