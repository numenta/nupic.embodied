"""A value function based on a GaussianMLP model."""
import torch
from torch import nn

from nupic.embodied.modules import GaussianMLPModule, GaussianDendriticMLPModule
from garage.torch.value_functions.value_function import ValueFunction
from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from torch.distributions import Normal

class GaussianMLPValueFunction(ValueFunction):
    """Gaussian MLP Value Function with Model.

    It fits the input data to a gaussian distribution estimated by
    a MLP.

    Args:
        env_spec (EnvSpec): Environment specification.
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
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): The name of the value function.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 layer_normalization=False,
                 name='GaussianMLPValueFunction'):
        super(GaussianMLPValueFunction, self).__init__(env_spec, name)

        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1

        self.module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=None,
            max_std=None,
            std_parameterization='exp',
            layer_normalization=layer_normalization)

    def compute_loss(self, obs, returns):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        dist = self.module(obs)
        ll = dist.log_prob(returns.reshape(-1, 1))
        loss = -ll.mean()
        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        return self.module(obs).mean.flatten(-2)


class GaussianDendriticValueFunction(ValueFunction):
    """Multiheaded MLP whose outputs are fed into a Normal distribution.
    A policy that contains a MLP to make prediction based on a gaussian
    distribution.
    Args:
        env_spec (EnvSpec): Environment specification.
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
    """

    def __init__(self,
                 env_spec,
                 dim_context,
                 num_tasks,
                 kw,
                 hidden_sizes=(32, 32),
                 num_segments=1,
                 kw_percent_on=0.05,
                 context_percent_on=1.0,
                 weight_sparsity=0.50,
                 weight_init="modified",
                 dendrite_init="modified",
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer,
                 output_nonlinearity=None,
                 preprocess_module_type=None,
                 preprocess_output_dim=128,
                 preprocess_kw_percent_on=0.1,
                 representation_module_type=None,
                 representation_module_dims=(128, 128),
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 ):
        super().__init__(env_spec, name='GaussianPolicy')

        self._obs_dim = env_spec.observation_space.flat_dim
        output_dim = 1

        self.module = GaussianDendriticMLPModule(
            input_dim=self._obs_dim - num_tasks,
            output_dim=output_dim,
            dim_context=dim_context,
            num_tasks=num_tasks,
            kw=kw,
            hidden_sizes=hidden_sizes,
            num_segments=num_segments,
            kw_percent_on=kw_percent_on,
            context_percent_on=context_percent_on,
            weight_sparsity=weight_sparsity,
            weight_init=weight_init,
            dendrite_init=dendrite_init,
            dendritic_layer_class=dendritic_layer_class,
            output_nonlinearity=output_nonlinearity,
            preprocess_module_type=preprocess_module_type,
            preprocess_output_dim=preprocess_output_dim,
            preprocess_kw_percent_on=preprocess_kw_percent_on,
            representation_module_type=representation_module_type,
            representation_module_dims=representation_module_dims,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            normal_distribution_cls=Normal
        )

    def compute_loss(self, obs, returns):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        dist = self.module(obs)
        ll = dist.log_prob(returns.reshape(-1, 1))
        loss = -ll.mean()
        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        return self.module(obs).mean.flatten(-2)
