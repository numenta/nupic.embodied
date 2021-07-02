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


# Code from https://github.com/qqadssp/Pytorch-Large-Scale-Curiosity
# and https://github.com/pathak22/exploration-by-disagreement

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
from nupic.embodied.disagreement import Trainer


def make_env_all_params(rank, args):
    """Initialize the environment and apply wrappers.

    Parameters
    ----------
    rank :
        Rank of the environment.
    args :
        Hyperparameters for this run.

    Returns
    -------
    env
        Environment with its individual wrappers.

    """
    if args.env_kind == "atari":
        from stable_baselines3.common.atari_wrappers import NoopResetEnv

        from nupic.embodied.disagreement.envs.wrappers import (
            AddRandomStateToInfo,
            ExtraTimeLimit,
            FrameStack,
            MaxAndSkipEnv,
            MontezumaInfoWrapper,
            ProcessFrame84,
            StickyActionEnv,
        )
        env = gym.make(args.env)
        assert "NoFrameskip" in env.spec.id
        if args.stickyAtari:
            env._max_episode_steps = args.max_episode_steps * 4
            env = StickyActionEnv(env)
        else:
            env = NoopResetEnv(env, noop_max=args.noop_max)
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        if not args.stickyAtari:
            env = ExtraTimeLimit(env, args.max_episode_steps)
        if "Montezuma" in args.env:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args.env_kind == "mario":
        from nupic.embodied.disagreement.envs.wrappers import make_mario_env
        env = make_mario_env()
    elif args.env_kind == "retro_multi":
        from nupic.embodied.disagreement.envs.wrappers import make_multi_pong
        env = make_multi_pong()
    elif args.env_kind == "roboarm":
        from real_robots.envs import REALRobotEnv

        from nupic.embodied.disagreement.envs.wrappers import CartesianControlDiscrete
        env = REALRobotEnv(objects=3, action_type="cartesian")
        env = CartesianControlDiscrete(
            env,
            crop_obs=args.crop_obs,
            repeat=args.act_repeat,
            touch_reward=args.touch_reward,
            random_force=args.random_force,
        )
        if args.resize_obs > 0:
            env = ResizeObservation(env, args.resize_obs)

    print("adding monitor")
    env = Monitor(env, filename=None)
    return env


if __name__ == "__main__":

    """
    Example usage:
        run the default environment (Breakout) in debugging mode (no wandb logging)
        python run.py--exp_name=ExperimentName --debugging

        run an experiment with random features and with logging:
        python run.py --exp_name=ExperimentName

        load a model locally and continue training:
        python run.py --exp_name=ExperimentName --load

        load a model from wandb artifact:
        python run.py --exp_name=ExperimentName --load
        --download_model_from=vclay/embodiedAI/ExperimentName:v0
    """

    # Parse from command line
    cmd_parser = create_cmd_parser()
    run_args = cmd_parser.parse_args()
    if "exp_name" not in run_args:
        cmd_parser.print_help()
        exit(1)

    # Parse from config dictionary
    exp_parser = create_exp_parser()
    exp_args = exp_parser.parse_dict(CONFIGS[run_args.exp_name])
    logging_args, env_args, trainer_args, learner_args = exp_args

    # Option to give a new wandb run name to the same experiment settings
    if run_args.wandb_run_name != "":
        run_args.wandb_run_name = run_args.exp_name

    print("Setting up Environment.")

    make_env = partial(make_env_all_params, args=env_args)

    print("Initializing Trainer.")

    if torch.cuda.is_available() and not run_args.cpu:
        print("GPU detected")
        dev_name = "cuda:0"
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("no GPU detected, using CPU instead.")
        dev_name = "cpu"
    device = torch.device(dev_name)
    print("device: " + str(device))

    # Verify if checkpoint dir is given and define dir where model will be savedd
    if "CHECKPOINT_DIR" not in os.environ:
        raise KeyError("Environment variable CHECKPOINT_DIR not found, required.")
    else:
        checkpoint_dir = os.path.join(os.environ["CHECKPOINT_DIR"], "robot_arm")

    

    trainer = Trainer(
        # TODO: should we set exp_name used in wandb.artifact() to wandb_run_name?
        exp_name=run_args.exp_name,
        make_env=make_env,
        device=device,
        logging_args=logging_args,
        env_args=env_args,
        trainer_args=trainer_args,
        learner_args=learner_args,
        debugging=run_args.debugging
    )

    if len(run_args.load) > 0 and len(run_args.download_model_from) == 0:
        # If the model is not downloaded from wandb we want to load the checkpoint first
        # to get the previous wandb_id to resume logging with that.
        trainer.load_models(debugging=run_args.debugging, model_dir=os.path.join(checkpoint_dir, run_args.load))
        logging_args.project_id = run_args.load.split('_')[-1]
        print("Using loaded Project ID " + logging_args.project_id)

    model_dir = os.path.join(
        checkpoint_dir, f"{run_args.exp_name}_{logging_args.project_id}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize wandb for logging (if not debugging)
    unrolled_config = {}
    for args in exp_args:
        for k, v in args.__dict__.items():
            unrolled_config[k] = v
    if not run_args.debugging:
        # TODO: Resume wandb logging is not working
        run = wandb.init(
            project="embodiedAI",
            name=run_args.exp_name + logging_args.project_id,
            id=trainer.wandb_id,
            group=logging_args.group,
            notes=logging_args.notes,
            config=unrolled_config,
            resume="allow",
        )
        if wandb.run.resumed:
            print(
                f"resuming wandb logging at step {wandb.run.step}"
                f" with run id {wandb.run.id}"
            )
        trainer.wandb_run = run

        model_stats_log_freq = 10
        wandb.watch(
            trainer.policy.features_model, log="all", log_freq=model_stats_log_freq
        )
        wandb.watch(trainer.policy.pd_hidden, log="all", log_freq=model_stats_log_freq)
        wandb.watch(trainer.policy.pd_head, log="all", log_freq=model_stats_log_freq)
        wandb.watch(trainer.policy.vf_head, log="all", log_freq=model_stats_log_freq)
        # Just log parameter & gradients of one dynamics net to avoid clutter.
        wandb.watch(
            trainer.dynamics_list[0].dynamics_net,
            log="all",
            log_freq=model_stats_log_freq,
        )

        if logging_args.detailed_wandb_logging:
            wandb.watch(
                trainer.feature_extractor.features_model,
                log="all",
                log_freq=model_stats_log_freq,
            )

            for i in range(trainer_args.num_dynamics):
                wandb.watch(
                    trainer.dynamics_list[i].dynamics_net,
                    log="all",
                    log_freq=model_stats_log_freq,
                )

    # TODO: debugging is used accross all code - is there a better way of defining it
    # instead of message passing from one object to another? Maybe as a global constant?

    if len(run_args.load) > 0 and len(run_args.download_model_from) > 0:
        trainer.load_models(
            debugging=run_args.debugging,
            download_model_from=run_args.download_model_from,
            model_dir=model_dir
        )

    try:
        trainer.train(debugging=run_args.debugging)
        print("Experiment finished training.")
    except KeyboardInterrupt:
        print("Training interrupted.")
        trainer.save_models(
            debugging=run_args.debugging,
            final=True,
            model_dir=model_dir
        )
        if not run_args.debugging:
            run.finish()
