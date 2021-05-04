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
from garage.torch.algos import TD3
import torch
import torch.nn.functional as F
from garage.torch import (global_device)
from nupic.torch.modules.sparse_weights import rezero_weights

class SparseWeightsTD3(TD3):

    def _optimize_policy(self, samples_data, grad_step_timer):
        """Perform algorithm optimization.

        Args:
            samples_data (dict): Processed batch data.
            grad_step_timer (int): Iteration number of the gradient time
                taken in the env.

        Returns:
            float: Loss predicted by the q networks
                (critic networks).
            float: Q value (min) predicted by one of the
                target q networks.
            float: Q value (min) predicted by one of the
                current q networks.
            float: Loss predicted by the policy
                (action network).

        """
        rewards = samples_data['rewards'].to(global_device()).reshape(-1, 1)
        terminals = samples_data['terminals'].to(global_device()).reshape(
            -1, 1)
        actions = samples_data['actions'].to(global_device())
        observations = samples_data['observations'].to(global_device())
        next_observations = samples_data['next_observations'].to(
            global_device())

        next_inputs = next_observations
        inputs = observations
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(
                -self._policy_noise_clip, self._policy_noise_clip)
            next_actions = (self._target_policy(next_inputs) + noise).clamp(
                -self._max_action, self._max_action)

            # Compute the target Q value
            target_Q1 = self._target_qf_1(next_inputs, next_actions)
            target_Q2 = self._target_qf_2(next_inputs, next_actions)
            target_q = torch.min(target_Q1, target_Q2)
            target_Q = rewards * self._reward_scaling + (
                1. - terminals) * self._discount * target_q

        # Get current Q values
        current_Q1 = self._qf_1(inputs, actions)
        current_Q2 = self._qf_2(inputs, actions)
        current_Q = torch.min(current_Q1, current_Q2)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize critic
        self._qf_optimizer_1.zero_grad()
        self._qf_optimizer_2.zero_grad()
        critic_loss.backward()
        self._qf_optimizer_1.step()
        self._qf_optimizer_2.step()
        self._qf_1.apply(rezero_weights)
        self._qf_2.apply(rezero_weights)

        # Deplay policy updates
        if grad_step_timer % self._update_actor_interval == 0:
            # Compute actor loss
            actions = self.policy(inputs)
            self._actor_loss = -self._qf_1(inputs, actions).mean()

            # Optimize actor
            self._policy_optimizer.zero_grad()
            self._actor_loss.backward()
            self._policy_optimizer.step()
            self.policy.apply(rezero_weights)

            # update target networks
            self._update_network_parameters()

        return (critic_loss.detach(), target_Q, current_Q.detach(),
                self._actor_loss.detach())

    def _evaluate_policy(self):
        self.policy.eval()
        result = super(SparseWeightsTD3, self)._evaluate_policy()
        self.policy.train()
        return result
