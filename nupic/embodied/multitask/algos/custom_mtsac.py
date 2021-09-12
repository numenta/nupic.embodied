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
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import wandb
from garage import StepType
from garage.torch.algos.mtsac import MTSAC
from time import time

from nupic.embodied.multitask.algos.custom_sac import CustomSAC
from nupic.embodied.utils.garage_utils import log_multitask_performance
from nupic.torch.modules.sparse_weights import rezero_weights

import logging


class CustomMTSAC(MTSAC, CustomSAC):
    def __init__(
        self,
        policy,
        qf1,
        qf2,
        replay_buffer,
        env_spec,
        sampler,
        train_task_sampler,
        *,
        num_tasks,
        gradient_steps_per_itr,
        max_episode_length_eval=None,
        fixed_alpha=None,
        target_entropy=None,
        initial_log_entropy=0.,
        discount=0.99,
        buffer_batch_size=64,
        min_buffer_size=10000,
        target_update_tau=5e-3,
        policy_lr=3e-4,
        qf_lr=3e-4,
        reward_scale=1.0,
        optimizer=torch.optim.Adam,
        steps_per_epoch=1,
        num_evaluation_episodes=5,
        task_update_frequency=1,
        wandb_logging=True,
        evaluation_frequency=25,
        fp16=False
    ):

        super().__init__(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            env_spec=env_spec,
            sampler=sampler,
            train_task_sampler=train_task_sampler,
            num_tasks=num_tasks,
            gradient_steps_per_itr=gradient_steps_per_itr,
            max_episode_length_eval=max_episode_length_eval,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            initial_log_entropy=initial_log_entropy,
            discount=discount,
            buffer_batch_size=buffer_batch_size,
            min_buffer_size=min_buffer_size,
            target_update_tau=target_update_tau,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            num_evaluation_episodes=num_evaluation_episodes,
            task_update_frequency=task_update_frequency,
            # TODO: remove test_sampler if overriding MTSAC or updating garage
            test_sampler=None
        )
        # Added samplers as local attributes since required in methods defined below
        self._train_task_sampler = train_task_sampler
        self._wandb_logging = wandb_logging
        self._evaluation_frequency = evaluation_frequency
        self._fp16 = fp16

        self._gs_qf1 = GradScaler()
        self._gs_qf2 = GradScaler()
        self._gs_policy = GradScaler()
        self._gs_alpha = GradScaler()

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        # Note: is a separate eval env really required?
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()

        last_return = None

        for epoch in trainer.step_epochs():
            for step in range(self._steps_per_epoch):
                t0 = time()
                if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
                    batch_size = None
                else:
                    batch_size = int(self._min_buffer_size)

                # Note: copying the policy to CPU - why? because collection is on CPU?
                with torch.no_grad():
                    agent_update = copy.deepcopy(self.policy).to("cpu")

                env_updates = None

                if step % self._task_update_frequency:
                    self._curr_train_tasks = self._train_task_sampler.sample(
                        self._num_tasks
                    )
                    env_updates = self._curr_train_tasks

                trainer.step_episode = trainer.obtain_samples(
                    trainer.step_itr, batch_size,
                    agent_update=agent_update,
                    env_update=env_updates
                )

                path_returns = []
                for path in trainer.step_episode:
                    self.replay_buffer.add_path(dict(
                        observation=path["observations"],
                        action=path["actions"],
                        reward=path["rewards"].reshape(-1, 1),
                        next_observation=path["next_observations"],
                        terminal=np.array([
                            step_type == StepType.TERMINAL
                            for step_type in path["step_types"]
                        ]).reshape(-1, 1)
                    ))
                    path_returns.append(sum(path["rewards"]))

                assert len(path_returns) == len(trainer.step_episode), \
                    "The number of path returns have to match number step episodes"

                self.episode_rewards.append(np.mean(path_returns))

                t1 = time()
                logging.debug("TRAINING DEVICE: ", next(self.policy.parameters()).device)
                # Note: why only one gradient step with all the data?
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()

            t2 = time()

            # logging at each epoch
            log_dict = self._log_statistics(policy_loss, qf1_loss, qf2_loss)
            if self._wandb_logging:
                log_dict["TotalEnvSteps"] = trainer.total_env_steps
            trainer.step_itr += 1

            # logging only when evaluating
            if epoch % self._evaluation_frequency == 0:
                last_return, eval_log_dict = self._evaluate_policy(trainer.step_itr)
                if log_dict is not None:
                    log_dict.update(eval_log_dict)

            t3 = time()

            if log_dict is not None:
                wandb.log(log_dict)

            t4 = time()

            # TODO: switch to logger.debug once logger is fixed
            logging.warn(f"Time to collect samples: {t1-t0:.2f}")
            logging.warn(f"Time to update gradient: {t2-t1:.2f}")
            logging.warn(f"Time to evaluate policy: {t3-t2:.2f}")
            logging.warn(f"Time to log:             {t4-t3:.2f}")

        return np.mean(last_return)
    
    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Raises:
            ValueError: If the number of tasks, num_tasks passed to
                this algorithm doesn't match the length of the task
                one-hot id in the observation vector.
        Returns:
            torch.Tensor: log_alpha. shape is (1, self.buffer_batch_size)
        """
        obs = samples_data["observation"]
        log_alpha = self._log_alpha
        one_hots = obs[:, -self._num_tasks:]

        if (log_alpha.shape[0] != one_hots.shape[1]
                or one_hots.shape[1] != self._num_tasks
                or log_alpha.shape[0] != self._num_tasks):
            raise ValueError(
                "The number of tasks in the environment does "
                "not match self._num_tasks. Are you sure that you passed "
                "The correct number of tasks?")
        
        with autocast(enabled=self._fp16):
            return torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()

    def _temperature_objective(self, log_pi, samples_data):
        """Compute the temperature/alpha coefficient loss.
        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.
        """
        alpha_loss = 0

        with autocast(enabled=self._fp16):
            if self._use_automatic_entropy_tuning:
                alpha_loss = (-(self._get_log_alpha(samples_data)) *
                            (log_pi.detach() + self._target_entropy)).mean()

            return alpha_loss

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        """Compute the Policy/Actor loss.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions (torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions (torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: loss from the Policy/Actor.
        """
        obs = samples_data["observation"]

        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        with autocast(enabled=self._fp16):
            min_q_new_actions = torch.min(self._qf1(obs, new_actions), 
                                          self._qf2(obs, new_actions))

            policy_objective = ((alpha * log_pi_new_actions) -
                                min_q_new_actions.flatten()).mean()

            return policy_objective

    def _critic_objective(self, samples_data):
        """Compute the Q-function/critic loss.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.
        """
        obs = samples_data["observation"]
        actions = samples_data["action"]
        rewards = samples_data["reward"].flatten()
        terminals = samples_data["terminal"].flatten()
        next_obs = samples_data["next_observation"]

        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        with autocast(enabled=self._fp16):
            q1_pred = self._qf1(obs, actions)
            q2_pred = self._qf2(obs, actions)

            new_next_actions_dist = self.policy(next_obs)[0]
            new_next_actions_pre_tanh, new_next_actions = (
                new_next_actions_dist.rsample_with_pre_tanh_value())
            new_log_pi = new_next_actions_dist.log_prob(value=new_next_actions, 
                                                        pre_tanh_value=new_next_actions_pre_tanh)

            target_q_values = torch.min(
                self._target_qf1(next_obs, new_next_actions),
                self._target_qf2(next_obs, new_next_actions)).flatten() - (alpha * new_log_pi)

            with torch.no_grad():
                q_target = rewards * self._reward_scale + (
                    1. - terminals) * self._discount * target_q_values
                    
            qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
            qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

            return qf1_loss, qf2_loss

    def optimize_policy(self, samples_data):
        """Optimize the policy q_functions, and temperature coefficient. Rezero
        model weights (if applicable) after each optimizer step.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        if not self._fp16:
            return super().optimize_policy(samples_data)

        obs = samples_data["observation"]
        
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        self._gs_qf1.scale(qf1_loss).backward()
        self._gs_qf1.step(self._qf1_optimizer)
        self._gs_qf1.update()
        self._qf1.apply(rezero_weights)

        self._qf2_optimizer.zero_grad()
        self._gs_qf2.scale(qf2_loss).backward()
        self._gs_qf2.step(self._qf2_optimizer)
        self._gs_qf2.update()
        self._qf2.apply(rezero_weights)

        with autocast():
            action_dists = self.policy(obs)[0]
            new_actions_pre_tanh, new_actions = (action_dists.rsample_with_pre_tanh_value())
            log_pi_new_actions = action_dists.log_prob(value=new_actions, 
                                                       pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)
        
        self._policy_optimizer.zero_grad()
        self._gs_policy.scale(policy_loss).backward()
        self._gs_policy.step(self._policy_optimizer)
        self._gs_policy.update()
        self.policy.apply(rezero_weights)

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)

            self._alpha_optimizer.zero_grad()
            self._gs_alpha.scale(alpha_loss).backward()
            self._gs_alpha.step(self._alpha_optimizer)
            self._gs_alpha.update()

        return policy_loss, qf1_loss, qf2_loss

    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        self.policy.eval()

        t00 = time()

        # Note: why is the policy on CPU?
        with torch.no_grad():
            agent_update = copy.deepcopy(self.policy).to("cpu")

        t01 = time()

        eval_batch = self._sampler.obtain_exact_episodes(
            n_eps_per_worker=self._num_evaluation_episodes,
            agent_update=agent_update,
        )

        t02 = time()

        last_return, log_dict = log_multitask_performance(
            epoch, eval_batch, self._discount, use_wandb=self._wandb_logging
        )

        t03 = time()

        logging.debug(f"Time to copy network to CPU (in evaluate pol): {t01-t00:.2f}")
        logging.debug(f"Time to obtain episodes (in evaluate pol):     {t02-t01:.2f}")
        logging.debug(f"Time to log results (in evaluate pol):         {t03-t02:.2f}")

        self.policy.train()
        return last_return, log_dict


    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        log_dict = None
        if self._wandb_logging:
            log_dict = {}
            with torch.no_grad():
                log_dict["AlphaTemperature/mean"] = self._log_alpha.exp().mean().item()
            log_dict["Policy/Loss"] = policy_loss.item()
            log_dict["QF/{}".format("Qf1Loss")] = float(qf1_loss)
            log_dict["QF/{}".format("Qf2Loss")] = float(qf2_loss)
            log_dict["ReplayBuffer/buffer_size"] = self.replay_buffer.n_transitions_stored  # noqa: E501
            log_dict["Average/TrainAverageReturn"] = np.mean(self.episode_rewards)

        return log_dict
