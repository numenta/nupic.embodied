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
"""MTSAC implementation based on Metaworld. Benchmarked on metaworld_experiments.
https://arxiv.org/pdf/1910.10897.pdf

Requires following environment variables: WANDB_DIR, CHECKPOINT_DIR, WANDB_API_KEY
"""

import click
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
from nupic.embodied.utils.garage_utils import (
    create_policy_net,
    create_qf_net,
    get_params,
)

t0 = time()


@click.command()
@click.option("--experiment_name")
@click.option("--config_pth")
@click.option("--seed", "seed", type=int, default=1)
@click.option("--timesteps", type=int, default=15000000)
@click.option("--use_wandb", default="True")
@click.option("--wandb_project_name", default="mt10")
@click.option("--gpu", default=None)
@wrap_experiment(snapshot_mode="none", name_parameters="passed", name="test_run",
                 log_dir=os.environ["CHECKPOINT_DIR"] or None, archive_launch_repo=False)
def mtsac_metaworld_mt10(
    ctxt=None, *, experiment_name, config_pth, seed, timesteps, use_wandb, wandb_project_name, gpu
):
    """Train MTSAC with metaworld_experiments environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).
        timesteps (int): Number of timesteps to run.
    """
    print(f"Initiation took {time()-t0:.2f} secs")

    # Get experiment parameters (e.g. hyperparameters) and save the json file
    params = get_params(config_pth)

    with open(ctxt.snapshot_dir + "/params.json", "w") as json_file:
        json.dump(params, json_file)

    if use_wandb == "True":
        use_wandb = True
        wandb.init(
            name=experiment_name,
            project=wandb_project_name,
            group="Baselines{}".format("mt10"),
            reinit=True,
            config=params,
        )
    else:
        use_wandb = False

    num_tasks = 10
    timesteps = timesteps
    deterministic.set_seed(seed)
    trainer = CustomTrainer(ctxt)
    mt10 = metaworld.MT10()

    train_task_sampler = MetaWorldTaskSampler(mt10, "train", add_env_onehot=True)

    assert num_tasks % 10 == 0, "Number of tasks have to divisible by 10"
    assert num_tasks <= 500, "Number of tasks should be less or equal 500"
    mt10_train_envs = train_task_sampler.sample(num_tasks)
    env = mt10_train_envs[0]()

    params["net"]["policy_min_std"] = np.exp(params["net"]["policy_min_log_std"])
    params["net"]["policy_max_std"] = np.exp(params["net"]["policy_max_log_std"])

    policy = create_policy_net(env_spec=env.spec, net_params=params["net"])
    qf1 = create_qf_net(env_spec=env.spec, net_params=params["net"])
    qf2 = create_qf_net(env_spec=env.spec, net_params=params["net"])

    replay_buffer = PathBuffer(
        capacity_in_transitions=int(params["general_setting"]["num_buffer_transitions"])
    )
    max_episode_length = env.spec.max_episode_length
    # Note: are the episode length the same among all tasks?

    sampler = RaySampler(
        agents=policy,
        envs=mt10_train_envs,
        max_episode_length=max_episode_length,
        # 1 sampler worker for each environment
        n_workers=num_tasks,
        worker_class=DefaultWorker
    )

    test_sampler = RaySampler(
        agents=policy,
        envs=mt10_train_envs,
        max_episode_length=max_episode_length,
        # 1 sampler worker for each environment
        n_workers=num_tasks,
        worker_class=EvalWorker
    )

    # Note:  difference between sampler and test sampler is only the worker
    # difference is one line in EvalWorker, uses average: a = agent_info['mean']
    # can we create a unified worker that contains both rules?

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
        gradient_steps_per_itr=int(max_episode_length * params["training"]["num_grad_steps_scale"]),
        num_tasks=num_tasks,
        min_buffer_size=max_episode_length * num_tasks,
        target_update_tau=params["training"]["target_update_tau"],
        discount=params["general_setting"]["discount"],
        buffer_batch_size=params["training"]["buffer_batch_size"],
        policy_lr=params["training"]["policy_lr"],
        qf_lr=params["training"]["qf_lr"],
        reward_scale=params["training"]["reward_scale"],
        num_evaluation_episodes=params["general_setting"]["eval_episodes"],
        task_update_frequency=params["training"]["task_update_frequency"],
        wandb_logging=use_wandb,
        evaluation_frequency=params["general_setting"]["evaluation_frequency"]
    )

    if gpu is not None:
        set_gpu_mode(True, gpu)

    mtsac.to()
    trainer.setup(algo=mtsac, env=mt10_train_envs)
    trainer.train(n_epochs=epochs, batch_size=steps_between_updates)


if __name__ == "__main__":
    print("Starting script...")
    mtsac_metaworld_mt10()
