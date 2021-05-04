#!/usr/bin/env python3
"""This is an example to train PPO on MT10 environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import wandb

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import MetaWorldTaskSampler
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler, DefaultWorker
from nupic.embodied.algos.custom_mt_ppo import CustomMTPPO
from garage.trainer import Trainer
from nupic.embodied.utils.utils import get_params, create_policy_net, create_vf_net


@click.command()
@click.option('--experiment_name')
@click.option('--config_pth')
@click.option('--seed', default=1)
@click.option('--n_workers', default=10)
@click.option('--n_tasks', default=10)
@click.option('--use_wandb', default='True')
@click.option('--wandb_username', default='avelu')
@wrap_experiment(snapshot_mode='none')
def mtppo_metaworld_mt10(ctxt, experiment_name, config_pth, seed, n_workers, n_tasks, use_wandb,
                         wandb_username):
    """Set up environment and algorithm and run the task.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.
        n_workers (int): The number of workers the sampler should use.
        n_tasks (int): Number of tasks to use. Should be a multiple of 10.
    """
    params = get_params(config_pth)
    set_seed(seed)
    mt10 = metaworld.MT10()
    train_task_sampler = MetaWorldTaskSampler(mt10,
                                              'train',
                                              lambda env, _: normalize(env),
                                              add_env_onehot=True)

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

    assert n_tasks % 10 == 0
    assert n_tasks <= 500

    envs = [env_up() for env_up in train_task_sampler.sample(n_tasks)]
    env = envs[0]

    policy = create_policy_net(env_spec=env.spec,
                               net_params=params["net"]
                               )
    value_function = create_vf_net(env_spec=env.spec,
                                   net_params=params["net"]
                                   )

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=n_workers,
                         worker_class=DefaultWorker)

    algo = CustomMTPPO(env_spec=env.spec,
                       policy=policy,
                       value_function=value_function,
                       sampler=sampler,
                       train_task_sampler=train_task_sampler,
                       num_tasks=n_tasks,
                       task_update_frequency=params["training"]["task_update_frequency"],
                       num_eval_eps=params["general_setting"]["eval_episodes"],
                       policy_lr=params["training"]["policy_lr"],
                       vf_lr=params["training"]["vf_lr"],
                       ppo_eps=params["training"]["ppo_eps"],
                       minibatch_size=params["training"]["minibatch_size"],
                       ppo_epochs=params["training"]["ppo_epochs"],
                       num_train_per_epoch=params["training"]["num_train_per_epoch"],
                       discount=params["general_setting"]["discount"],
                       gae_lambda=params["training"]["gae_lambda"],
                       center_adv=False,
                       wandb_logging=use_wandb,
                       eval_freq=params["general_setting"]["eval_freq"],
                       stop_entropy_gradient=True,
                       entropy_method='max',

                       )

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=params["training"]["epochs"],
                  batch_size=params["training"]["batch_episodes_per_task"],
                  plot=False)


if __name__ == '__main__':
    mtppo_metaworld_mt10()
