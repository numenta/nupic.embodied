from garage.trainer import Trainer
from dowel import logger
import time
import os


class CustomTrainer(Trainer):
    # custom trainer class which removes Tabular usage
    def log_diagnostics(self, pause_for_plot=False):
        """Log diagnostics.

        Args:
            pause_for_plot (bool): Pause for plot.

        """
        logger.log('Time %.2f s' % (time.time() - self._start_time))
        logger.log('EpochTime %.2f s' % (time.time() - self._itr_start_time))
        logger.log('TotalEnvSteps: ' + str(self._stats.total_env_steps))

        if self._plot:
            self._plotter.update_plot(self._algo.policy,
                                      self._algo.max_episode_length)
            if pause_for_plot:
                input('Plotting evaluation run: Press Enter to " "continue...')

    def step_epochs(self):
        """Step through each epoch.

        This function returns a magic generator. When iterated through, this
        generator automatically performs services such as snapshotting and log
        management. It is used inside train() in each algorithm.

        The generator initializes two variables: `self.step_itr` and
        `self.step_episode`. To use the generator, these two have to be
        updated manually in each epoch, as the example shows below.

        Yields:
            int: The next training epoch.

        Examples:
            for epoch in trainer.step_epochs():
                trainer.step_episode = trainer.obtain_samples(...)
                self.train_once(...)
                trainer.step_itr += 1

        """
        self._start_time = time.time()
        self.step_itr = self._stats.total_itr
        self.step_episode = None

        # Used by integration tests to ensure examples can run one epoch.
        n_epochs = int(
            os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                           self._train_args.n_epochs))

        logger.log('Obtaining samples...')

        for epoch in range(self._train_args.start_epoch, n_epochs):
            self._itr_start_time = time.time()
            with logger.prefix('epoch #%d | ' % epoch):
                yield epoch
                save_episode = (self.step_episode
                                if self._train_args.store_episodes else None)

                self._stats.last_episode = save_episode
                self._stats.total_epoch = epoch
                self._stats.total_itr = self.step_itr

                self.save(epoch)

                if self.enable_logging:
                    self.log_diagnostics(self._train_args.pause_for_plot)
                    logger.dump_all(self.step_itr)
