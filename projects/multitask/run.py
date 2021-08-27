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

import os
from functools import partial
from parser import create_cmd_parser, create_exp_parser

import gym
import torch
import wandb
from gym.wrappers import ResizeObservation
# TODO: check if Monitor from stable_baselines3 is same as baselines Monitor
from stable_baselines3.common.monitor import Monitor

from experiments import CONFIGS

import logging
import sys

import json
import metaworld
import numpy as np
import os
import wandb

from garage import wrap_experiment
from garage.experiment import deterministic
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker, EvalWorker, RaySampler
from garage.torch import set_gpu_mode
from nupic.embodied.multitask.custom_trainer import CustomTrainer
from time import time

from nupic.embodied.multitask.algos.custom_mtsac import CustomMTSAC
from nupic.embodied.utils.parser_utils import merge_args
from nupic.embodied.utils.garage_utils import (
    create_policy_net,
    create_qf_net,
    get_params,
)

import inspect

t0 = time()


def init_experiment(
    ctxt=None,
    *,
    experiment_name,
    use_wandb,
    project_name,
    experiment_args,
    training_args,
    network_args,
    use_gpu
):
    """Train MTSAC with metaworld_experiments environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        gpu (int): The ID of the gpu to be used (used on multi-gpu machines).
        timesteps (int): Number of timesteps to run.
    """

    num_tasks = network_args.num_tasks
    timesteps = experiment_args.timesteps
    deterministic.set_seed(experiment_args.seed)

    if use_wandb:
        wandb.init(
            name=experiment_name,
            project=project_name,
            group="Baselines{}".format(num_tasks),
            reinit=True,
            config=merge_args(
                (experiment_args, training_args, network_args)
            ),
        )

    trainer = CustomTrainer(ctxt)

    # Note: different classes whether it uses 10 or 50 tasks. Why?
    mt_env = metaworld.MT10() if num_tasks <= 10 else metaworld.MT50()

    train_task_sampler = MetaWorldTaskSampler(
        mt_env, "train", add_env_onehot=True
    )

    # TODO: add some clarifying comments of why these asserts are required
    assert num_tasks % 10 == 0, "Number of tasks have to divisible by 10"
    assert num_tasks <= 500, "Number of tasks should be less or equal 500"
    mt_train_envs = train_task_sampler.sample(num_tasks)
    env = mt_train_envs[0]()

    policy = create_policy_net(env_spec=env.spec, net_params=network_args)
    qf1 = create_qf_net(env_spec=env.spec, net_params=network_args)
    qf2 = create_qf_net(env_spec=env.spec, net_params=network_args)

    replay_buffer = PathBuffer(
        capacity_in_transitions=training_args.num_buffer_transitions
    )
    max_episode_length = env.spec.max_episode_length
    # Note: are the episode length the same among all tasks?

    sampler = RaySampler(
        agents=policy,
        envs=mt_train_envs,
        max_episode_length=max_episode_length,
        # 1 sampler worker for each environment
        n_workers=num_tasks,
        worker_class=DefaultWorker
    )

    test_sampler = RaySampler(
        agents=policy,
        envs=mt_train_envs,
        max_episode_length=max_episode_length,
        # 1 sampler worker for each environment
        n_workers=num_tasks,
        worker_class=EvalWorker
    )

    # Note:  difference between sampler and test sampler is only the worker
    # difference is one line in EvalWorker, uses average: a = agent_info['mean']
    # TODO: can we create a unified worker that contains both rules?

    # Number of transitions before a set of gradient updates
    steps_between_updates = int(max_episode_length * num_tasks)

    # epoch: 1 cycle of data collection + gradient updates
    epochs = timesteps // steps_between_updates

    mtsac = CustomMTSAC(
        env_spec=env.spec,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        replay_buffer=replay_buffer,
        sampler=sampler,
        train_task_sampler=train_task_sampler,
        test_sampler=test_sampler,
        gradient_steps_per_itr=int(
            max_episode_length * training_args.num_grad_steps_scale
        ),
        num_tasks=num_tasks,
        min_buffer_size=max_episode_length * num_tasks,
        target_update_tau=training_args.target_update_tau,
        discount=training_args.discount,
        buffer_batch_size=training_args.buffer_batch_size,
        policy_lr=training_args.policy_lr,
        qf_lr=training_args.qf_lr,
        reward_scale=training_args.reward_scale,
        num_evaluation_episodes=training_args.eval_episodes,
        task_update_frequency=training_args.task_update_frequency,
        wandb_logging=use_wandb,
        evaluation_frequency=training_args.evaluation_frequency
    )

    # TODO: do we have to fix which GPU to use?
    if use_gpu:
        set_gpu_mode(True, 0)

    # TODO: add a clarifying comment of what mtsac.to is doing
    mtsac.to()
    trainer.setup(algo=mtsac, env=mt_train_envs)

    if experiment_args.do_train:
        trainer.train(n_epochs=epochs, batch_size=steps_between_updates)

    # Debug mode returns all main classes for inspection
    if experiment_args.debug_mode:
        return trainer, env, policy, qf1, qf2, replay_buffer, mtsac


if __name__ == "__main__":

    """
    Example usage:
        python run.py -e experiment_name
    """

    # Parse from command line
    cmd_parser = create_cmd_parser()
    run_args = cmd_parser.parse_args()
    if "exp_name" not in run_args:
        cmd_parser.print_help()
        exit(1)

    # Parse from config dictionary
    exp_parser = create_exp_parser()
    all_args = exp_parser.parse_dict(CONFIGS[run_args.exp_name])
    logging_args, experiment_args, training_args, network_args = all_args

    # Setup logging based on verbose defined by the user
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if run_args.verbose else logging.INFO
    )

    # Gives an additional option to define a wandb run name
    if run_args.wandb_run_name != "":
        run_args.wandb_run_name = run_args.exp_name

    # Automatically detects whether or not to use GPU
    use_gpu = torch.cuda.is_available() and not run_args.cpu
    logging.info(f"Using GPU: {use_gpu}")

    wrapped_init_experiment = wrap_experiment(
        function=init_experiment,
        log_dir=logging_args.log_dir,
        snapshot_mode=logging_args.snapshot_mode,
        snapshot_gap=logging_args.snapshot_gap,
        name_parameters="passed",
        archive_launch_repo=False,
    )

    wrapped_init_experiment(
        experiment_name=run_args.exp_name,
        use_wandb=not run_args.local_only,
        use_gpu=use_gpu,
        project_name=logging_args.project_name,
        experiment_args=experiment_args,
        training_args=training_args,
        network_args=network_args,
    )
