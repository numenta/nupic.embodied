#!/usr/bin/env python3
"""MTSAC implementation based on Metaworld. Benchmarked on MT10.
https://arxiv.org/pdf/1910.10897.pdf
"""
import click
import metaworld
import numpy as np

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker, EvalWorker, RaySampler
from garage.torch import set_gpu_mode
from nupic.embodied.algos.custom_mtsac import CustomMTSAC
from garage.trainer import Trainer
from nupic.embodied.utils.garage_utils import get_params, create_policy_net, create_qf_net
import json
import wandb


@click.command()
@click.option('--experiment_name')
@click.option('--config_pth')
@click.option('--seed', 'seed', type=int, default=1)
@click.option('--use_wandb', default='True')
@click.option('--wandb_username', default='avelu')
@click.option('--gpu', default=None)
@wrap_experiment(snapshot_mode='none', name_parameters="passed", name="test_run",
                 log_dir="/Users/avelu/nta/data/test_run")
def mtsac_metaworld_mt10(ctxt=None, *, experiment_name, config_pth, seed, use_wandb, wandb_username, gpu):
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

    # get experiment parameters (e.g. hyperparameters) and save the json file
    params = get_params(config_pth)

    with open(ctxt.snapshot_dir + '/params.json', 'w') as json_file:
        json.dump(params, json_file)

    if use_wandb == 'True':
        use_wandb = True
        wandb.init(
            name=experiment_name,
            entity=wandb_username,
            project="mt10",
            group="Baselines{}".format("mt10"),
            reinit=True,
            config=params,
        )
    else:
        use_wandb = False

    n_tasks = 10
    # timesteps = 500000000
    timesteps = 15000000
    deterministic.set_seed(seed)
    trainer = Trainer(ctxt)
    mt10 = metaworld.MT10()  # pylint: disable=no-member

    # pylint: disable=missing-return-doc, missing-return-type-doc
    train_task_sampler = MetaWorldTaskSampler(mt10,
                                              'train',
                                              add_env_onehot=True)

    assert n_tasks % 10 == 0
    assert n_tasks <= 500
    mt10_train_envs = train_task_sampler.sample(n_tasks)
    env = mt10_train_envs[0]()

    params["net"]["policy_min_std"] = np.exp(params["net"]["policy_min_log_std"])
    params["net"]["policy_max_std"] = np.exp(params["net"]["policy_max_log_std"])

    policy = create_policy_net(env_spec=env.spec,
                               net_params=params["net"]
                               )
    qf1 = create_qf_net(env_spec=env.spec,
                        net_params=params["net"]
                        )
    qf2 = create_qf_net(env_spec=env.spec,
                        net_params=params["net"]
                        )

    replay_buffer = PathBuffer(capacity_in_transitions=int(params["general_setting"]["num_buffer_transitions"]), )
    num_tasks = 10
    max_episode_length = env.spec.max_episode_length

    sampler = RaySampler(
        agents=policy,
        envs=mt10_train_envs,
        max_episode_length=max_episode_length,
        # 1 sampler worker for each environment
        n_workers=num_tasks,
        worker_class=DefaultWorker)

    test_sampler = RaySampler(
        agents=policy,
        envs=mt10_train_envs,
        max_episode_length=max_episode_length,
        # 1 sampler worker for each environment
        n_workers=num_tasks,
        worker_class=EvalWorker)

    batch_size = int(max_episode_length * num_tasks) # number of transitions before a set of gradient updates
    num_evaluation_points = timesteps // batch_size # number of times policy is evaluated (also the # of epochs)
    epochs = timesteps // batch_size
    # number of times new batch of samples + gradient updates are done per epoch
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    mtsac = CustomMTSAC(env_spec=env.spec,
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
                        wandb_logging=use_wandb
                        )
    if gpu is not None:
        set_gpu_mode(True, gpu)
    mtsac.to()
    trainer.setup(algo=mtsac, env=mt10_train_envs)
    trainer.train(n_epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    # pylint: disable=missing-kwoa
    mtsac_metaworld_mt10()
