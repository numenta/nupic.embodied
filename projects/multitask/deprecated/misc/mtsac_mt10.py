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
"""MTSAC implementation based on Metaworld. Benchmarked on MT10.
https://arxiv.org/pdf/1910.10897.pdf
"""
import click
import metaworld
import numpy as np
import ray
from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import (
    FragmentWorker,
    LocalSampler,
    MultiprocessingSampler,
    RaySampler,
    VecWorker,
    WorkerFactory,
)
from garage.torch import set_gpu_mode
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
from torch import nn
from torch.nn import functional as F

from nupic.embodied.multitask.samplers.single_vecworker_sampler import SingleVecWorkSampler


@click.command()
@click.option('--seed', 'seed', type=int, default=1)
@click.option('--gpu', '_gpu', type=int, default=None)
@click.option('--n_tasks', default=10)
@click.option('--timesteps', default=20000000)
@wrap_experiment(snapshot_mode='none')
def mtsac_metaworld_mt10(ctxt=None, *, seed, _gpu, n_tasks, timesteps):
    """Train MTSAC with MT10 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).
        n_tasks (int): Number of tasks to use. Should be a multiple of 10.
        timesteps (int): Number of timesteps to run.
    """
    deterministic.set_seed(seed)
    trainer = Trainer(ctxt)
    mt10 = metaworld.MT10()  # pylint: disable=no-member
    mt10_test = metaworld.MT10()  # pylint: disable=no-member

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def wrap(env, _):
        return normalize(env, normalize_reward=True)

    train_task_sampler = MetaWorldTaskSampler(mt10,
                                              'train',
                                              wrap,
                                              add_env_onehot=True)
    test_task_sampler = MetaWorldTaskSampler(mt10_test,
                                             'train',
                                             add_env_onehot=True)
    assert n_tasks % 10 == 0
    assert n_tasks <= 500
    mt10_train_envs = train_task_sampler.sample(n_tasks)
    env = mt10_train_envs[0]()
    mt10_test_envs = [env_up() for env_up in test_task_sampler.sample(n_tasks)]

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[400, 400, 400],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )
    meta_batch_size = 10

    # ray.init(local_mode=True, log_to_driver=False, ignore_reinit_error=True)
    sampler = SingleVecWorkSampler(
        agents=policy,
        envs=mt10_train_envs,
        n_workers=meta_batch_size,
        max_episode_length=env.spec.max_episode_length,
        # 1 sampler worker for each environment
        worker_class=FragmentWorker,
        # increasing n_envs increases the vectorization of a sampler worker
        # which improves runtime performance, but you will need to adjust this
        # depending on your memory constraints. For reference, each worker by
        # default uses n_envs=8. Each environment is approximately ~50mb large
        # so creating 50 envs with 8 copies comes out to 20gb of memory. Many
        # users want to be able to run multiple seeds on 1 machine, so I have
        # reduced this to n_envs = 2 for 2 copies in the meantime.
        worker_args=dict(n_envs=10))

    # one episode for each task between gradient steps
    batch_size = int(env.spec.max_episode_length * meta_batch_size)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  sampler=sampler,
                  gradient_steps_per_itr=env.spec.max_episode_length,
                  eval_env=mt10_test_envs,
                  env_spec=env.spec,
                  num_tasks=10,
                  steps_per_epoch=epoch_cycles,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1500,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=1280)
    if _gpu is not None:
        set_gpu_mode(True, _gpu)
    trainer.setup(algo=mtsac, env=mt10_train_envs)
    import time
    s = time.time()
    mtsac.to()
    print(time.time() - s)

    s = time.time()
    # trainer.step_episode = trainer.obtain_samples(0, 1500, None, None)
    trainer.step_episode = trainer.obtain_samples(0, 2000, None, None)
    print((time.time() - s))
    a = 2

    from garage import StepType
    path_returns = []
    for path in trainer.step_episode:
        mtsac.replay_buffer.add_path(
            dict(observation=path['observations'],
                 action=path['actions'],
                 reward=path['rewards'].reshape(-1, 1),
                 next_observation=path['next_observations'],
                 terminal=np.array([
                     step_type == StepType.TERMINAL
                     for step_type in path['step_types']
                 ]).reshape(-1, 1)))
        path_returns.append(sum(path['rewards']))

    s = time.time()
    for _ in range(10):
        trainer._algo.train_once()
    print((time.time() - s) / 10)

    # trainer.train(n_epochs=epochs, batch_size=batch_size)


# pylint: disable=missing-kwoa
if __name__ == '__main__':
    mtsac_metaworld_mt10()
