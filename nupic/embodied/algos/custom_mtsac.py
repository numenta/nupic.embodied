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
from garage.torch.algos.mtsac import MTSAC
from nupic.embodied.algos.custom_sac import CustomSAC
import torch
import numpy as np
import copy
from nupic.embodied.utils.garage_utils import log_multitask_performance
from dowel import tabular
from garage import StepType

class CustomMTSAC(MTSAC, CustomSAC):
    def __init__(
        self,
        policy,
        qf1,
        qf2,
        replay_buffer,
        env_spec,
        sampler,
        test_sampler,
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
        min_buffer_size=int(1e4),
        target_update_tau=5e-3,
        policy_lr=3e-4,
        qf_lr=3e-4,
        reward_scale=1.0,
        optimizer=torch.optim.Adam,
        steps_per_epoch=1,
        num_evaluation_episodes=5,
        task_update_frequency=1,
        wandb_logging=True
    ):

        super(CustomMTSAC, self).__init__(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            env_spec=env_spec,
            sampler=sampler,
            test_sampler=test_sampler,
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
            task_update_frequency=task_update_frequency
        )
        self._wandb_logging = wandb_logging

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for _ in trainer.step_epochs():
            for i in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                with torch.no_grad():
                    agent_update = copy.deepcopy(self.policy).to("cpu")
                # import ipdb; ipdb.set_trace()
                env_updates = None
                if (not i and not i % self._task_update_frequency) or (self._task_update_frequency == 1):
                    self._curr_train_tasks = self._train_task_sampler.sample(self._num_tasks)
                    env_updates = self._curr_train_tasks
                trainer.step_episode = trainer.obtain_samples(
                    trainer.step_itr, batch_size, agent_update=agent_update,
                    env_update=env_updates)
                path_returns = []
                for path in trainer.step_episode:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=np.array([
                                 step_type == StepType.TERMINAL
                                 for step_type in path['step_types']
                             ]).reshape(-1, 1)))
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) == len(trainer.step_episode)
                self.episode_rewards.append(np.mean(path_returns))
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()
            last_return = self._evaluate_policy(trainer.step_itr)
            log_dict = self._log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)

            if self._wandb_logging:
                log_dict['TotalEnvSteps'] = trainer.total_env_steps

            trainer.step_itr += 1

        return np.mean(last_return)

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
        with torch.no_grad():
            agent_update = copy.deepcopy(self.policy).to("cpu")
        eval_batch = self._test_sampler.obtain_exact_episodes(
            n_eps_per_worker=self._num_evaluation_episodes,
            agent_update=agent_update)
        last_return = log_multitask_performance(epoch, eval_batch,
                                                self._discount, use_wandb=self._wandb_logging)
        self.policy.train()
        return last_return

    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        with torch.no_grad():
            tabular.record('AlphaTemperature/mean',
                           self._log_alpha.exp().mean().item())
        tabular.record('Policy/Loss', policy_loss.item())
        tabular.record('QF/{}'.format('Qf1Loss'), float(qf1_loss))
        tabular.record('QF/{}'.format('Qf2Loss'), float(qf2_loss))
        tabular.record('ReplayBuffer/buffer_size',
                       self.replay_buffer.n_transitions_stored)
        tabular.record('Average/TrainAverageReturn',
                       np.mean(self.episode_rewards))

        log_dict = None
        if self._wandb_logging:
            log_dict = {}
            with torch.no_grad():
                log_dict['AlphaTemperature/mean'] = self._log_alpha.exp().mean().item()
            log_dict['Policy/Loss'] = policy_loss.item()
            log_dict['QF/{}'.format('Qf1Loss')] = float(qf1_loss)
            log_dict['QF/{}'.format('Qf2Loss')] = float(qf2_loss)
            log_dict['ReplayBuffer/buffer_size'] = self.replay_buffer.n_transitions_stored
            log_dict['Average/TrainAverageReturn'] = np.mean(self.episode_rewards)

        return log_dict