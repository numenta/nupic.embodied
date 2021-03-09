import abc
import torch
from torch import nn
from torch.distributions.independent import Independent
from torch.distributions import Normal
from garage.torch.distributions import TanhNormal


class GaussianBaseModule(nn.Module):
    def __init__(self,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 normal_distribution_cls=Normal
                 ):
        super(GaussianBaseModule, self).__init__()
        self._learn_std = learn_std
        self._init_std = init_std
        self._min_std = min_std
        self._max_std = max_std
        self._std_parameterization = std_parameterization
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

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        # reset the instance variables after buffers are moved to device
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
