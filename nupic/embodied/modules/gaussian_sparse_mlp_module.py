"""GaussianMLPModule."""
import abc

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal
from nupic.embodied.utils.models import MultiHeadedSparseMLP
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule


class GaussianSparseMLPBaseModule(nn.Module):
    """Base of Sparse MLP for Gaussian Output.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        learn_std (bool):GaussianSparseMLPBaseModule Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
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
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 output_nonlinearity=None,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_output_nonlinearity=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 linear_activity_percent_on=(0.1, 0.1),
                 linear_weight_percent_on=(0.4, 0.1),
                 boost_strength=1.67,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 consolidated_sparse_weights=False,
                 normal_distribution_cls=Normal,
                 ):
        super().__init__()

        assert len(hidden_sizes) == len(std_hidden_sizes) == len(linear_activity_percent_on) == len(linear_weight_percent_on)
        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_parameterization = std_parameterization
        self._output_nonlinearity = output_nonlinearity
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        init_std_param = torch.Tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

        # sparse layer information
        self._linear_activity_percent_on = linear_activity_percent_on
        self._linear_weight_percent_on = linear_weight_percent_on
        self._boost_strength = boost_strength
        self._boost_strength_factor = boost_strength_factor
        self._duty_cycle_period = duty_cycle_period
        self._k_inference_factor = k_inference_factor
        self._use_batch_norm = use_batch_norm
        self._dropout = dropout
        self._consolidated_sparse_weights = consolidated_sparse_weights

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']
        self._min_std_param = buffers['min_std_param']
        self._max_std_param = buffers['max_std_param']

    @abc.abstractmethod
    def _get_mean_and_log_std(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.

        """
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()
        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist


class GaussianSparseMLPTwoHeadedModule(GaussianSparseMLPBaseModule):
    """GaussianMLPModule which has only one mean network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
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
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 output_nonlinearity=None,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):

        super(GaussianSparseMLPTwoHeadedModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             output_nonlinearity=output_nonlinearity,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)

        self._shared_mean_log_std_network = MultiHeadedSparseMLP(
            input_size=input_dim,
            output_dims=(self._action_dim, self._action_dim),
            output_nonlinearities=self._output_nonlinearity,
            hidden_sizes=self._hidden_sizes,
            linear_activity_percent_on=self._linear_activity_percent_on,
            linear_weight_percent_on=self._linear_weight_percent_on,
            boost_strength=self._boost_strength,
            boost_strength_factor=self._boost_strength_factor,
            duty_cycle_period=self._duty_cycle_period,
            k_inference_factor=self._k_inference_factor,
            use_batch_norm=self._use_batch_norm,
            dropout=self._dropout,
            consolidated_sparse_weights=self._consolidated_sparse_weights
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
