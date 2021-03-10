from nupic.embodied.models import MultiHeadedSparseMLP

class SparseMLP(MultiHeadedSparseMLP):
    """
    A dendritic network which is similar to a MLP with a two hidden layers, except that
    activations are modified by dendrites. The context input to the network is used as
    input to the dendritic weights.
    """

    def __init__(self, input_size,
                 output_dim,
                 output_nonlinearity=None,
                 hidden_sizes=(32, 32),
                 linear_activity_percent_on=(0.1, 0.1),
                 linear_weight_percent_on=(0.4, 0.4),
                 boost_strength=1.67,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 consolidated_sparse_weights=False,
                 ):

        super(SparseMLP, self).__init__(
            input_size=input_size,
            num_heads=1,
            output_dims=(output_dim),
            output_nonlinearities=(output_nonlinearity),
            hidden_sizes=hidden_sizes,
            linear_activity_percent_on=linear_activity_percent_on,
            linear_weight_percent_on=linear_weight_percent_on,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            k_inference_factor=k_inference_factor,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            consolidated_sparse_weights=consolidated_sparse_weights
        )
        self._output_dim = output_dim

    def forward(self, x):
        return super().forward(x)[0]

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim
