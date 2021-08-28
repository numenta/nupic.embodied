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

Requires following environment variables: WANDB_DIR, LOG_DIR, WANDB_API_KEY
"""

import click
import json
import metaworld
import numpy as np
import os
import wandb
import ray
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
# EvalWorker not available in latest version of garage, can't find a commit
# where it was available
from garage.sampler import DefaultWorker # , EvalWorker
from garage.torch import set_gpu_mode
from garage.trainer import Trainer
from time import time

from nupic.embodied.multitask.samplers.gpu_sampler import RaySampler
from nupic.embodied.multitask.algos.custom_mtsac import CustomMTSAC
from nupic.embodied.utils.garage_utils import (
    create_policy_net,
    create_qf_net,
    get_params,
)

from time import sleep

t0 = time()

# make sure the actors have been killed
ray.shutdown()

import gc

## MEM utils ##
def mem_report():
    """Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported"""

    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: "CPU" or "GPU" in current implementation """
        print("Storage on %s" %(mem_type))
        print("-" * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print("%s\t\t%s\t\t%.2f" % (
                element_type,
                size,
                mem) )
        print("-"*LEN)
        print("Total Tensors: %d \tUsed Memory Space: %.2f MBytes" % (total_numel, total_mem) )
        print("-"*LEN)

    LEN = 65
    print("="*LEN)
    objects = gc.get_objects()
    print("%s\t%s\t\t\t%s" %("Element type", "Size", "Used MEM(MBytes)") )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, "GPU")
    _mem_report(host_tensors, "CPU")
    print("="*LEN)


@click.command()
@click.option("-e", "--experiment_name")
@click.option("-c", "--config_pth")
@click.option("-s", "--seed", "seed", type=int, default=1)
@click.option("-w", "--use_wandb", default="True")
@click.option("-g", "--gpu", is_flag=True, default=False)
@wrap_experiment(snapshot_mode="none", name_parameters="passed", name="test_run",
                 log_dir=os.environ["CHECKPOINT_DIR"] or None, archive_launch_repo=False)
def mtsac_metaworld_mt10(
    ctxt=None, *, experiment_name, config_pth, seed, use_wandb, gpu
):
    """Train MTSAC with MT10 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).
        timesteps (int): Number of timesteps to run.
    """
    print(f"Initiation took {time()-t0:.2f} secs")

    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f"Using GPU: {gpu}, Device: {device}")

    # maybe overring other things - this is required, why?
    if gpu:
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)

    # Get experiment parameters (e.g. hyperparameters) and save the json file
    params = get_params(config_pth)

    with open(ctxt.snapshot_dir + "/params.json", "w") as json_file:
        json.dump(params, json_file)

    if use_wandb == "True":
        use_wandb = True
        wandb.init(
            name=experiment_name,
            project="mt10_debug",
            group="Baselines{}".format("mt10"),
            reinit=True,
            config=params,
        )
    else:
        use_wandb = False

    num_tasks = params["net"]["num_tasks"]
    timesteps = 15000000
    deterministic.set_seed(seed)
    trainer = Trainer(ctxt)

    # Note: different classes whether it uses 10 or 50 tasks. Why?
    if num_tasks <= 10:
        mt_env = metaworld.MT10()
    else:
        mt_env = metaworld.MT50()

    train_task_sampler = MetaWorldTaskSampler(mt_env, "train", add_env_onehot=True)

    assert num_tasks % 10 == 0, "Number of tasks have to divisible by 10"
    assert num_tasks <= 500, "Number of tasks should be less or equal 500"
    mt_train_envs = train_task_sampler.sample(num_tasks)
    env = mt_train_envs[0]()

    params["net"]["policy_min_std"] = np.exp(params["net"]["policy_min_log_std"])
    params["net"]["policy_max_std"] = np.exp(params["net"]["policy_max_log_std"])

    policy = create_policy_net(env_spec=env.spec, net_params=params["net"])
    print("Created policy")

    qf1 = create_qf_net(env_spec=env.spec, net_params=params["net"])
    qf2 = create_qf_net(env_spec=env.spec, net_params=params["net"])
    print("Created value functions")

    replay_buffer = PathBuffer(
        capacity_in_transitions=int(params["general_setting"]["num_buffer_transitions"])
    )
    max_episode_length = env.spec.max_episode_length
    # Note: are the episode length the same among all tasks?

    sampler = RaySampler(
        agents=policy,
        envs=mt_train_envs,
        max_episode_length=max_episode_length,
        cpus_per_worker=params["sampler"]["cpus_per_worker"],
        gpus_per_worker=params["sampler"]["gpus_per_worker"],
        seed=None,  # set to get_seed() to make it deterministic
    )

    # will probably still need the sampler
    test_sampler = sampler
    # test_sampler = RaySampler(
    #     agents=policy,
    #     envs=mt_train_envs,
    #     max_episode_length=max_episode_length,
    #     # 1 sampler worker for each environment
    #     n_workers=num_tasks,
    #     worker_class=EvalWorker
    # )

    # Note:  difference between sampler and test sampler is only the worker
    # difference is one line in EvalWorker, uses average: a = agent_info["mean"]
    # can we create a unified worker that cointais both rules?

    # Number of transitions before a set of gradient updates
    # Note: should we use avg episode length, if they are not same for all tasks?
    batch_size = int(max_episode_length * num_tasks)

    # TODO: this whole block seems unnecessary, it is not doing anything.
    # Number of times policy is evaluated (also the # of epochs)
    num_evaluation_points = timesteps // batch_size
    epochs = timesteps // batch_size
    # number of times new batch of samples + gradient updates are done per epoch
    epoch_cycles = epochs // num_evaluation_points  # this will always be equal to 1
    epochs = epochs // epoch_cycles

    mtsac = CustomMTSAC(
        env_spec=env.spec,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        replay_buffer=replay_buffer,
        sampler=sampler,
        train_task_sampler=train_task_sampler,
        test_sampler=test_sampler,
        gradient_steps_per_itr=1,
        num_tasks=num_tasks,
        steps_per_epoch=epoch_cycles,
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
        evaluation_frequency=params["general_setting"]["evaluation_frequency"],
    )

    print("Created algo")

    mtsac.to(device=device)
    print("Moved networks to device")

    trainer.setup(algo=mtsac, env=mt_train_envs)
    print("Setup trainer")

    trainer.train(n_epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    print("Starting script...")
    mtsac_metaworld_mt10()
