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
import abc
import copy

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent
import torch.nn.functional as F

from garage.torch.distributions import TanhNormal
from garage.torch.modules import MultiHeadedMLPModule

from nupic.embodied.multitask.modules import CustomDendriticMLP
from garage.torch import NonLinearity

class GaussianMLPBaseModule(nn.Module):
    """Base of GaussianMLPModel. Adapted from:
    https://github.com/rlworkgroup/garage/blob/668b83392ffeae14882a36f0cab8c40a2b9d11b3/src/garage/torch/modules/gaussian_mlp_module.py
    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and then applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        learn_std (bool): Is std trainable.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes,
        hidden_nonlinearity,
        output_nonlinearity,
        min_std,
        max_std,
        normal_distribution_cls,
        init_std,
        std_parameterization,
        hidden_w_init,
        hidden_b_init,
        output_w_init,
        output_b_init,
        layer_normalization,
        learn_std
    ):
        super().__init__()

        self._init_std = torch.Tensor([init_std])

        log_std = torch.Tensor([init_std] * output_dim)

        if learn_std:
            self._log_std = torch.nn.Parameter(log_std)
        else:
            self._log_std = log_std
            self.register_buffer("log_std", self.log_std)

        self._min_std = torch.Tensor([min_std])
        self.register_buffer("min_std", self._min_std)

        self._max_std = torch.Tensor([max_std])
        self.register_buffer("max_std", self._max_std)

        self._std_parameterization = std_parameterization
        self._normal_distribution = normal_distribution_cls

        assert self._std_parameterization in {"exp", "softplus"}


    def to(self, *args, **kwargs):
        """Move the module to the specified device.
        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.
        """
        super().to(*args, **kwargs)

        buffers = dict(self.named_buffers())

        if not isinstance(self._log_std, torch.nn.Parameter):
            self._log_std = buffers["log_std"]

        self._min_std = buffers["min_std"]
        self._max_std = buffers["max_std"]

    @abc.abstractmethod
    def get_mean_log_std(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.
        Args:
            *inputs: Input to the module.
        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.
        """
        mean, log_std_uncentered = self.get_mean_log_std(*inputs)

        log_std_uncentered = log_std_uncentered.clamp(min=self._min_std.item(),
                                                      max=self._max_std.item())

        if self._std_parameterization == "exp":
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()

        dist = self._normal_distribution(mean, std)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist


class GaussianTwoHeadedMLPModule(GaussianMLPBaseModule):
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
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and then applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        learn_std (bool): Is std trainable.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes,
        hidden_nonlinearity,
        output_nonlinearity,
        min_std,
        max_std,
        normal_distribution_cls,
        init_std,
        std_parameterization,
        hidden_w_init,
        hidden_b_init,
        output_w_init,
        output_b_init,
        layer_normalization,
        learn_std
    ):

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            min_std=min_std,
            max_std=max_std,
            normal_distribution_cls=normal_distribution_cls,
            init_std=init_std,
            std_parameterization=std_parameterization,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
            learn_std=learn_std,
        )

        self.mean_log_std = CustomMultiHeadedMLPModule(
            n_heads=2,
            input_dim=input_dim,
            output_dims=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearities=output_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_w_inits=output_w_init,
            output_b_inits=[
                nn.init.zeros_,
                lambda x: nn.init.constant_(x, self._init_std.item())
            ],
            layer_normalization=layer_normalization
        )

    def get_mean_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.
        Args:
            *inputs: Input to the module.
        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.
        """
        return self.mean_log_std(*inputs)


class GaussianTwoHeadedDendriticMLPModule(GaussianMLPBaseModule):
    def __init__(
        self,
        input_dim,
        context_dim,
        output_dim,
        hidden_sizes,
        layers_modulated,
        num_segments,
        kw_percent_on,
        context_percent_on,
        weight_sparsity,
        weight_init,
        dendrite_weight_sparsity,
        dendrite_init,
        dendritic_layer_class,
        output_nonlinearity,
        preprocess_module_type,
        preprocess_output_dim,
        preprocess_kw_percent_on,
        min_std,
        max_std,
        normal_distribution_cls,
        init_std,
        std_parameterization,
        layer_normalization,
        learn_std
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=None,
            output_nonlinearity=output_nonlinearity,
            min_std=min_std,
            max_std=max_std,
            normal_distribution_cls=normal_distribution_cls,
            init_std=init_std,
            std_parameterization=std_parameterization,
            hidden_w_init=None,
            hidden_b_init=None,
            output_w_init=None,
            output_b_init=None,
            layer_normalization=layer_normalization,
            learn_std=learn_std
        )

        self.mean_log_std = CustomDendriticMLP(
            input_dim=input_dim,
            context_dim=context_dim,
            output_sizes=(output_dim, output_dim),
            hidden_sizes=hidden_sizes,
            layers_modulated=layers_modulated,
            num_segments=num_segments,
            kw_percent_on=kw_percent_on,
            context_percent_on=context_percent_on,
            weight_sparsity=weight_sparsity,
            weight_init=weight_init,
            dendrite_weight_sparsity=dendrite_weight_sparsity,
            dendrite_init=dendrite_init,
            dendritic_layer_class=dendritic_layer_class,
            output_nonlinearity=output_nonlinearity,
            preprocess_module_type=preprocess_module_type,
            preprocess_output_dim=preprocess_output_dim,
            preprocess_kw_percent_on=preprocess_kw_percent_on,
        )

    def get_mean_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.
        Args:

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.
        """
        return self.mean_log_std(*inputs)


class CustomMultiHeadedMLPModule(nn.Module):
    """MultiHeadedMLPModule Model.

    A PyTorch module composed only of a multi-layer perceptron (MLP) with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        n_heads (int): Number of different output layers
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 n_heads,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 hidden_layer_normalization=None,
                 hidden_layer_normalization_affine=False,
                ):
        super().__init__()

        self._layers = nn.ModuleList()

        output_dims = self._check_parameter_for_output_layer(
            "output_dims", output_dims, n_heads)
        output_w_inits = self._check_parameter_for_output_layer(
            "output_w_inits", output_w_inits, n_heads)
        output_b_inits = self._check_parameter_for_output_layer(
            "output_b_inits", output_b_inits, n_heads)
        output_nonlinearities = self._check_parameter_for_output_layer(
            "output_nonlinearities", output_nonlinearities, n_heads)

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module("linear", linear_layer)

            if hidden_layer_normalization == "batch_normalization":
                hidden_layers.add_module(
                    "batch_normalization",
                    nn.BatchNorm1d(size, affine=hidden_layer_normalization_affine)
                )
            elif hidden_layer_normalization == "layer_normalization":
                hidden_layers.add_module(
                    "layer_normalization",
                    nn.LayerNorm(
                        size, elementwise_affine=hidden_layer_normalization_affine
                    )
                )

            if hidden_nonlinearity:
                hidden_layers.add_module("non_linearity",
                                         NonLinearity(hidden_nonlinearity))

            self._layers.append(hidden_layers)
            prev_size = size

        print("Running network with batch normalization")
        print(hidden_layers)
        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(prev_size, output_dims[i])
            output_w_inits[i](linear_layer.weight)
            output_b_inits[i](linear_layer.bias)
            output_layer.add_module("linear", linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module("non_linearity",
                                        NonLinearity(output_nonlinearities[i]))

            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            n_heads (int): number of head

        Returns:
            list: list of variables (length of n_heads)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_heads

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_heads
            if len(var) == n_heads:
                return var
            msg = ("{} should be either an integer or a collection of length "
                   "n_heads ({}), but {} provided.")
            raise ValueError(msg.format(var_name, n_heads, var))
        return [copy.deepcopy(var) for _ in range(n_heads)]

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


class MLPModule(CustomMultiHeadedMLPModule):
    """MLP Model.

    A Pytorch module composed only of a multi-layer perceptron (MLP), which
    maps real-valued inputs to real-valued outputs.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__(1, input_dim, output_dim, hidden_sizes,
                         hidden_nonlinearity, hidden_w_init, hidden_b_init,
                         output_nonlinearity, output_w_init, output_b_init,
                         layer_normalization)

        self._output_dim = output_dim

    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            torch.Tensor: Output value

        """
        return super().forward(input_value)[0]

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim
