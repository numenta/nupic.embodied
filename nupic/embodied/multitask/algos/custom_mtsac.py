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
import wandb
from garage import StepType
from garage.torch.algos.mtsac import MTSAC
from time import time

from nupic.embodied.multitask.algos.custom_sac import CustomSAC
from nupic.embodied.utils.garage_utils import log_multitask_performance


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
        wandb_logging=True,
        evaluation_frequency=25,
    ):

        super().__init__(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            env_spec=env_spec,
            sampler=sampler,
            # TODO: can't find parent class with compatible signature
            # test_sampler=test_sampler,
            # train_task_sampler=train_task_sampler,
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
        # Added samplers as local attributes since required in methods defined below
        self._test_sampler = test_sampler
        self._train_task_sampler = train_task_sampler
        self._wandb_logging = wandb_logging
        self._evaluation_frequency = evaluation_frequency

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
                print("TRAINING DEVICE: ", next(self.policy.parameters()).device)
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

            print(f"Time to collect samples: {t1-t0:.2f}")
            print(f"Time to update gradient: {t2-t1:.2f}")
            print(f"Time to evaluate policy: {t3-t2:.2f}")
            print(f"Time to log:             {t4-t3:.2f}")

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

        t00 = time()

        # Note: why is the policy on CPU?
        with torch.no_grad():
            agent_update = copy.deepcopy(self.policy).to("cpu")

        t01 = time()

        eval_batch = self._test_sampler.obtain_exact_episodes(
            n_eps_per_worker=self._num_evaluation_episodes,
            agent_update=agent_update
        )

        t02 = time()

        last_return, log_dict = log_multitask_performance(
            epoch, eval_batch, self._discount, use_wandb=self._wandb_logging
        )

        t03 = time()

        print(f"Time to copy network to CPU (in evaluate pol): {t01-t00:.2f}")
        print(f"Time to obtain episodes (in evaluate pol):     {t02-t01:.2f}")
        print(f"Time to log results (in evaluate pol):         {t03-t02:.2f}")

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
