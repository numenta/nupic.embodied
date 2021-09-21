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
import logging
import os
import time

import numpy as np
import wandb
from dowel import logger
from garage.experiment.experiment import dump_json
from garage.trainer import NotSetupError, TrainArgs, Trainer


class CustomTrainer(Trainer):
    """Custom trainer class which
    - removes Tabular usage
    - refactors bidirectional message in run epoch intro traditional OOP architecture
    """

    def log_diagnostics(self, pause_for_plot=False):
        """Log diagnostics.

        Args:
            pause_for_plot (bool): Pause for plot.

        """
        logging.info("Time %.2f s" % (time.time() - self._start_time))
        logging.info("EpochTime %.2f s" % (time.time() - self._itr_start_time))
        logging.info("TotalEnvSteps: " + str(self._stats.total_env_steps))

        if self._plot:
            self._plotter.update_plot(self._algo.policy,
                                      self._algo.max_episode_length)
            if pause_for_plot:
                input("Plotting evaluation run: Press Enter to " "continue...")

    def train(
        self,
        n_epochs,
        batch_size=None,
        plot=False,
        store_episodes=False,
        pause_for_plot=False,
        wandb_logging=True,
        evaluation_frequency=25
    ):
        """
        Start training.
        This version replaces step_epochs in original garage Trainer

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int or None):
                Number of environment steps (samples) collected per epoch
                Only defines how many samples will be collected and added to buffer
                Does not define size of the batch used in the gradient update.

            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.

        Raises:
            NotSetupError: If train() is called before setup().

        Returns:
            float: The average return in last epoch cycle.

        """

        if not self._has_setup:
            raise NotSetupError(
                "Use setup() to setup trainer before training.")

        # Save arguments for restore
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot
        self._start_worker()

        # Log experiment json file
        log_dir = self._snapshotter.snapshot_dir
        summary_file = os.path.join(log_dir, "experiment.json")
        dump_json(summary_file, self)

        logging.info("Starting training...")
        self._start_time = time.time()
        self.step_episode = None
        returns = []

        # Loop through epochs and call algorithm to run one epoch at at ime
        for epoch in range(self._train_args.start_epoch, n_epochs):
            self._itr_start_time = time.time()

            # Run training epoch
            log_dict = self._algo.run_epoch(env_steps_per_epoch=batch_size)
            self.total_env_steps = log_dict["TotalEnvSteps"]  # TODO: needed?

            # Run evaluation, with a given frequency
            if epoch % evaluation_frequency == 0:
                eval_returns, eval_log_dict = self._algo._evaluate_policy(epoch)
                log_dict["average_return"] = np.mean(eval_returns)
                log_dict.update(eval_log_dict)
                returns.append(eval_returns)

            # Logging and updating state variables
            if wandb_logging:
                wandb.log(log_dict)
            self.log_epoch_legacy(epoch)  # for backwards compatibility garage logger
            # TODO: are these always equal if steps_per_epoch = 1? remove it
            self._stats.total_epoch += 1
            self._stats.total_itr += 1

        self._shutdown_worker()

        return np.mean(returns)

    def log_epoch_legacy(self, epoch):

        # TODO: control frequency of saving
        with logger.prefix("epoch #%d | " % epoch):
            save_episode = (self.step_episode
                            if self._train_args.store_episodes else None)

            self._stats.last_episode = save_episode
            self.save(epoch)

            # TODO: turn on and of logging through argument given to garage Trainer
            if self.enable_logging:
                self.log_diagnostics(self._train_args.pause_for_plot)
                logger.dump_all(epoch)
