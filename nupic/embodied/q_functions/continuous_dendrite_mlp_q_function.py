"""This modules creates a continuous Q-function network."""

import torch

from nupic.embodied.models import DendriticMLP


class ContinuousDendriteMLPQFunction(DendriticMLP):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, dim_context, **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        self._dim_context = dim_context
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim - self._dim_context
        self._action_dim = env_spec.action_space.flat_dim

        DendriticMLP.__init__(self,
                              input_size=self._obs_dim + self._action_dim,
                              output_dim=1,
                              dim_context=dim_context,
                              **kwargs)

    # pylint: disable=arguments-differ
    def forward(self, observations, actions):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.

        Returns:
            torch.Tensor: Output value
        """
        obs_portion, context_portion = observations[:, :self._obs_dim], observations[:, self._obs_dim:]
        return super().forward(torch.cat([obs_portion, actions], 1), context_portion)
