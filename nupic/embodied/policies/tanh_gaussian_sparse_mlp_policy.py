"""TanhGaussianMLPPolicy."""
import numpy as np

from garage.torch.distributions import TanhNormal
from nupic.embodied.modules.gaussian_sparse_mlp_module import GaussianSparseMLPTwoHeadedModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class TanhGaussianSparseMLPPolicy(StochasticPolicy):
    """Multiheaded MLP whose outputs are fed into a TanhNormal distribution.

    A policy that contains a MLP to make prediction based on a gaussian
    distribution with a tanh transformation.

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
                 hidden_sizes=(32, 32),
                 mean_nonlinearity=None,
                 std_nonlinearity=None,
                 linear_activity_percent_on=(0.1, 0.1),
                 linear_weight_percent_on=(0.4, 0.4),
                 boost_strength=1.67,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 consolidated_sparse_weights=False,
                 init_std=1.0,
                 min_std=np.exp(-20.),
                 max_std=np.exp(2.),
                 std_parameterization='exp'):
        super().__init__(env_spec, name='TanhGaussianPolicy')

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._module = GaussianSparseMLPTwoHeadedModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            mean_nonlinearity=mean_nonlinearity,
            std_nonlinearity=std_nonlinearity,
            linear_activity_percent_on=linear_activity_percent_on,
            linear_weight_percent_on=linear_weight_percent_on,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            k_inference_factor=k_inference_factor,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            consolidated_sparse_weights=consolidated_sparse_weights,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            normal_distribution_cls=TanhNormal
        )

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        dist = self._module(observations)
        ret_mean = dist.mean.cpu()
        ret_log_std = (dist.variance.sqrt()).log().cpu()
        return dist, dict(mean=ret_mean, log_std=ret_log_std)
