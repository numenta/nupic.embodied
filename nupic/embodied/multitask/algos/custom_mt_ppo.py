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

import numpy as np
import torch
import wandb

from dowel import tabular
from garage.np import discount_cumsum
from garage.torch import filter_valids, global_device, prefer_gpu, set_gpu_mode
from garage.torch._functions import np_to_torch
from garage.torch.algos import PPO
from garage.torch.optimizers import OptimizerWrapper

from nupic.embodied.utils.garage_utils import (
    compute_advantages,
    log_multitask_performance,
    log_performance,
)
from nupic.torch.modules.sparse_weights import rezero_weights


class CustomMTPPO(PPO):
    """Modified implemenetation of Proximal Policy Optimization (PPO) for Multi-Task settings.


    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_lr (float): policy optimizer learning rate.
        vf_lr (float): value function optimizer learning rate.
        ppo_eps (float): epsilon clipping term for PPO.
        minibatch_size (int): size of minibatches.
        ppo_epochs (int): number of times to iterate through a batch.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """
    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 sampler,
                 num_eval_eps=3,
                 policy_lr=2.5e-4,
                 vf_lr=2.5e-4,
                 ppo_eps=2e-1,
                 minibatch_size=64,
                 ppo_epochs=10,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=0.97,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 wandb_logging=True,
                 eval_freq=1,
                 multitask=True,
                 num_tasks=None,
                 task_update_frequency=1,
                 train_task_sampler=None,
                 gpu_training=False):

        policy_optimizer = OptimizerWrapper(
            (torch.optim.Adam, dict(lr=vf_lr)),
            policy,
            max_optimization_epochs=ppo_epochs,
            minibatch_size=minibatch_size)

        vf_optimizer = OptimizerWrapper(
            (torch.optim.Adam, dict(lr=policy_lr)),
            value_function,
            max_optimization_epochs=ppo_epochs,
            minibatch_size=minibatch_size)

        super(CustomMTPPO, self).__init__(
            env_spec=env_spec,
            policy=policy,
            value_function=value_function,
            sampler=sampler,
            policy_optimizer=policy_optimizer,
            vf_optimizer=vf_optimizer,
            lr_clip_range=ppo_eps,
            num_train_per_epoch=num_train_per_epoch,
            discount=discount,
            gae_lambda=gae_lambda,
            center_adv=center_adv,
            positive_adv=positive_adv,
            policy_ent_coeff=policy_ent_coeff,
            use_softplus_entropy=use_softplus_entropy,
            stop_entropy_gradient=stop_entropy_gradient,
            entropy_method=entropy_method
        )
        self._task_update_frequency = task_update_frequency
        self._multitask = multitask
        self._num_tasks = num_tasks
        self._train_task_sampler = train_task_sampler
        self._num_evaluation_episodes = num_eval_eps
        self._wandb_logging = wandb_logging
        self._eval_freq = eval_freq
        self._gpu_training = gpu_training

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for i, _ in enumerate(trainer.step_epochs()):
            if not self._multitask:
                trainer.step_path = trainer.obtain_episodes(trainer.step_itr)
            else:
                env_updates = None
                assert self._train_task_sampler is not None
                if (not i % self._task_update_frequency) or (self._task_update_frequency == 1):
                    env_updates = self._train_task_sampler.sample(self._num_tasks)
                trainer.step_path = self.obtain_exact_trajectories(trainer, env_update=env_updates)

            # do training on GPU
            if self._gpu_training:
                prefer_gpu()
                self.to(device=global_device())

            log_dict, last_return = self._train_once(trainer.step_itr, trainer.step_path)

            # move back to CPU for collection
            set_gpu_mode(False)
            self.to(device=global_device())

            if self._wandb_logging:
                # log dict should be a dict, not None
                log_dict['total_env_steps'] = trainer.total_env_steps
                wandb.log(log_dict)
            trainer.step_itr += 1

        return last_return

    def _train_once(self, itr, eps):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            eps (EpisodeBatch): A batch of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        obs = np_to_torch(eps.padded_observations)
        rewards = np_to_torch(eps.padded_rewards)
        returns = np_to_torch(
            np.stack([
                discount_cumsum(reward, self.discount)
                for reward in eps.padded_rewards
            ]))
        valids = eps.lengths
        with torch.no_grad():
            baselines = self._value_function(obs)

        if self._multitask:
            undiscounted_returns, log_dict = log_multitask_performance(itr, eps, discount=self._discount)
        else:
            undiscounted_returns, log_dict = log_performance(itr, eps, discount=self._discount)

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = np_to_torch(eps.observations)
        actions_flat = np_to_torch(eps.actions)
        rewards_flat = np_to_torch(eps.rewards)
        returns_flat = torch.cat(filter_valids(returns, valids))

        advs_flat = self._compute_advantage(rewards, valids, baselines)

        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_before = self._value_function.compute_loss(
                obs_flat, returns_flat)
            kl_before = self._compute_kl_constraint(obs)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat)

        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_after = self._value_function.compute_loss(
                obs_flat, returns_flat)
            kl_after = self._compute_kl_constraint(obs)
            policy_entropy = self._compute_policy_entropy(obs)

        if self._wandb_logging:
            # log_dict should not be None
            log_dict['Train/Policy/LossBefore'] = policy_loss_before.item()
            log_dict['Train/Policy/LossAfter'] = policy_loss_after.item()
            log_dict['Train/Policy/dLoss'] = (policy_loss_before - policy_loss_after).item()
            log_dict['Train/Policy/KLBefore'] = kl_before.item()
            log_dict['Train/Policy/KL'] = kl_after.item()
            log_dict['Train/Policy/Entropy'] = policy_entropy.mean().item()

            log_dict['Train/VF/LossBefore'] = vf_loss_before.item()
            log_dict['Train/VF/LossAfter'] = vf_loss_after.item()
            log_dict['Train/VF/dLoss'] = (vf_loss_before - vf_loss_after).item()

        with tabular.prefix(self.policy.name):
            tabular.record('/LossBefore', policy_loss_before.item())
            tabular.record('/LossAfter', policy_loss_after.item())
            tabular.record('/dLoss',
                           (policy_loss_before - policy_loss_after).item())
            tabular.record('/KLBefore', kl_before.item())
            tabular.record('/KL', kl_after.item())
            tabular.record('/Entropy', policy_entropy.mean().item())

        with tabular.prefix(self._value_function.name):
            tabular.record('/LossBefore', vf_loss_before.item())
            tabular.record('/LossAfter', vf_loss_after.item())
            tabular.record('/dLoss',
                           vf_loss_before.item() - vf_loss_after.item())

        self._old_policy.load_state_dict(self.policy.state_dict())
        return log_dict, np.mean(undiscounted_returns)

    def _train(self, obs, actions, rewards, returns, advs):
        r"""Train the policy and value function with minibatch.

        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.

        """
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, rewards, advs):
            self._train_policy(*dataset)
            self.policy.apply(rezero_weights)

        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)
            self._value_function.apply(rezero_weights)

    def _compute_advantage(self, rewards, valids, baselines):
        r"""Compute mean value of loss.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.

        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.

        """
        advantages = compute_advantages(self._discount, self._gae_lambda,
                                        self.max_episode_length, baselines,
                                        rewards)
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat

    def obtain_exact_trajectories(self, trainer, env_update=None):
        """Obtain an exact amount of trajs from each env being sampled from.
        Args:
            trainer (Trainer): Experiment trainer, which provides services
                such as snapshotting and sampler control.
            env_updates: updates to the env (i.e. rotate task)
        Returns:
            episodes (EpisodeBatch): Batch of episodes.
        """
        episodes_per_trajectory = trainer._train_args.batch_size
        agent_update = self.policy.get_param_values()
        sampler = trainer._sampler
        episodes = sampler.obtain_exact_episodes(
                              episodes_per_trajectory,
                              agent_update,
                              env_update=env_update)
        trainer._stats.total_env_steps += sum(episodes.lengths)
        return episodes

    @property
    def networks(self):
        return [self.policy, self._old_policy, self._value_function]

    def to(self, device=None):
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
