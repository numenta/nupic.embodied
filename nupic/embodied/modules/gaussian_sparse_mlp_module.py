"""GaussianMLPModule."""

import torch
from torch.distributions import Normal
from nupic.embodied.models import MultiHeadedSparseMLP
from nupic.embodied.modules.gaussian_base_module import GaussianBaseModule


class GaussianSparseMLPTwoHeadedModule(GaussianBaseModule):
    """GaussianMLPModule which has only one mean network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Action dimension to output.
        mean_nonlinearity (callable): Activation function for mean output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        std_nonlinearity (callable): Activation function for std output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        linear_activity_percent_on (list[float]): Percent of ON (non-zero) units
        linear_weight_percent_on (list[float]): Percent of weights that are allowed to be
                                       non-zero in the linear layer
        boost_strength (float): boost strength (0.0 implies no boosting)
        boost_strength_factor (float): Boost strength factor to use [0..1]
        duty_cycle_period (int): The period used to calculate duty cycles
        k_inference_factor (float): During inference (training=False) we increase
                                   `percent_on` in all sparse layers by this factor
        use_batch_norm (bool): whether to use batch norm
        dropout (float): dropout value
        consolidated_sparse_weights (bool): whether to use consolidated sparse weights
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 mean_nonlinearity=None,
                 std_nonlinearity=None,
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
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 normal_distribution_cls=Normal):

        super(GaussianSparseMLPTwoHeadedModule,
              self).__init__(learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             normal_distribution_cls=normal_distribution_cls)

        self._shared_mean_log_std_network = MultiHeadedSparseMLP(
            input_size=input_dim,
            num_heads=2,
            output_dims=(output_dim, output_dim),
            output_nonlinearities=(mean_nonlinearity, std_nonlinearity),
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

    def _get_mean_and_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return self._shared_mean_log_std_network(*inputs)
