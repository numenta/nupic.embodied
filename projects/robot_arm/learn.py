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

import argparse
import os
from functools import partial

import gym
import torch
import wandb
from gym.wrappers import ResizeObservation

from baselines.bench import Monitor
from trainer import Trainer


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
    if args["env_kind"] == "atari":
        from baselines.common.atari_wrappers import FrameStack, NoopResetEnv

        from nupic.embodied.envs.wrappers import (
            AddRandomStateToInfo,
            ExtraTimeLimit,
            MaxAndSkipEnv,
            MontezumaInfoWrapper,
            ProcessFrame84,
            StickyActionEnv,
        )
        env = gym.make(args["env"])
        assert "NoFrameskip" in env.spec.id
        if args["stickyAtari"]:
            env._max_episode_steps = args["max_episode_steps"] * 4
            env = StickyActionEnv(env)
        else:
            env = NoopResetEnv(env, noop_max=args["noop_max"])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        if not args["stickyAtari"]:
            env = ExtraTimeLimit(env, args["max_episode_steps"])
        if "Montezuma" in args["env"]:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == "mario":
        from nupic.embodied.envs.wrappers import make_mario_env
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        from nupic.embodied.envs.wrappers import make_multi_pong
        env = make_multi_pong()
    elif args["env_kind"] == "roboarm":
        from real_robots.envs import REALRobotEnv

        from nupic.embodied.envs.wrappers import CartesianControlDiscrete
        env = REALRobotEnv(objects=3, action_type="cartesian")
        env = CartesianControlDiscrete(
            env,
            crop_obs=args["crop_obs"],
            repeat=args["act_repeat"],
            touch_reward=args["touch_reward"],
            random_force=args["random_force"],
        )
        if args["resize_obs"] > 0:
            env = ResizeObservation(env, args["resize_obs"])

    print("adding monitor")
    env = Monitor(env, filename=None)
    return env

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Experiment Parameters:
    parser.add_argument("--exp_name", type=str, default="")
    # Extra specification for wandb logging
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--notes", type=str, default="")

    # Trainer parameters
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    # Normalize layer activations of the feature extractor
    parser.add_argument("--layernorm", type=int, default=0)
    # Whether to use the information of a state being terminal.
    parser.add_argument("--use_done", type=int, default=0)
    # Coefficients with which the internal and external rewards are multiplied
    parser.add_argument("--ext_coeff", type=float, default=0.0)
    parser.add_argument("--int_coeff", type=float, default=1.0)
    # Auxiliary task for feature encoding phi
    parser.add_argument(
        "--feat_learning",
        type=str,
        default="none",
        choices=["none", "idf", "vaesph", "vaenonsph"],
    )
    # Number of dynamics models
    parser.add_argument("--num_dynamics", type=int, default=5)
    # Weight of dynamic loss in the total loss
    parser.add_argument("--dyn_loss_weight", type=float, default=1.0)
    # Network parameters
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--policy_hidden_dim", type=int, default=512)
    # Whether to use the disagreement between dynamics models as internal reward
    parser.add_argument("--dont_use_disagreement", action="store_false", default=True)

    # Run options
    # Specify --load to load an existing model (needs to have same exp_name)
    parser.add_argument("--load", action="store_true", default=False)
    # Download a model from a wandb artifact (specify path)
    parser.add_argument("--download_model_from", type=str, default="")
    # option to use when debugging so not every test run is logged.
    parser.add_argument("--debugging", action="store_true", default=False)
    # frequencies in num_updates (not time_steps)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--model_save_freq", type=int, default=-1)

    # Environment parameters:
    parser.add_argument(
        "--env", help="environment ID", default="BreakoutNoFrameskip-v4", type=str
    )
    parser.add_argument("--max-episode-steps", default=4500, type=int)
    parser.add_argument("--env_kind", type=str, default="atari")
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--act_repeat", type=int, default=10)
    parser.add_argument("--stickyAtari", action="store_true", default=True)
    parser.add_argument("--crop_obs", action="store_true", default=True)
    parser.add_argument("--touch_reward", action="store_true", default=False)
    parser.add_argument("--random_force", action="store_true", default=False)
    # With current CNN possible sizes are 36, 44, 52, 60, 68, 76, 84, ..., 180
    parser.add_argument("--resize_obs", type=int, default=0)

    # Optimization parameters:
    parser.add_argument("--lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--nminibatches", type=int, default=8)
    parser.add_argument("--norm_adv", type=int, default=1)  # normalize advantages
    parser.add_argument("--norm_rew", type=int, default=1)  # normalize rewards
    parser.add_argument("--lr", type=float, default=1e-4)  # learning rate
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--nepochs", type=int, default=3)
    parser.add_argument("--num_timesteps", type=int, default=int(1024))

    # Rollout parameters:
    parser.add_argument("--nsteps_per_seg", type=int, default=128)
    parser.add_argument("--nsegs_per_env", type=int, default=1)
    parser.add_argument("--envs_per_process", type=int, default=128)
    parser.add_argument("--nlumps", type=int, default=1)
    parser.add_argument(
        "-b", "--backprop_through_reward", action="store_true", default=False
    )
    return parser


if __name__ == "__main__":

    """
    Example usage:
        run the default environment (Breakout) in debugging mode (no wandb logging)
        python learn.py --envs_per_process=8 --num_timesteps=1000 --debugging

        run an experiment with random features and with logging:
        python learn.py -- env=BreakoutNoFrameskip-v4 --envs_per_process=8
        --num_timesteps=10000 --feat_learning=None --exp_name=ExperimentName
        --group=Implementation --notes="Some notes about this run"

        load a model locally and continue training:
        python learn.py --num_timesteps=10000 --exp_name=ExperimentName --load

        load a model from wandb artifact:
        python learn.py --num_timesteps=10000 --exp_name=ExperimentName --load
        --download_model_from=vclay/embodiedAI/ExperimentName:v0
    """

    parser = create_parser()
    args = parser.parse_args()

    print("Setting up Environment.")

    make_env = partial(make_env_all_params, args=args.__dict__)

    print("Initializing Trainer.")

    if torch.cuda.is_available():
        print("GPU detected")
        dev_name = "cuda:0"
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("no GPU detected, using CPU instead.")
        dev_name = "cpu"
    device = torch.device(dev_name)
    print("device: " + str(device))

    trainer = Trainer(
        make_env=make_env,
        num_timesteps=args.num_timesteps,
        hyperparameter=args.__dict__,
        envs_per_process=args.envs_per_process,
        num_dynamics=args.num_dynamics,
        use_disagreement=args.dont_use_disagreement,
        device=device,
    )

    if args.load and len(args.download_model_from) == 0:
        # If the model is not downloaded from wandb we want to load the checkpoint first
        # to get the previous wandb_id to resume logging with that.
        trainer.load_models()

    # Initialize wandb for logging (if not debugging)
    if not args.debugging:
        run = wandb.init(
            project="embodiedAI",
            name=args.exp_name,
            id=trainer.wandb_id,
            group=args.group,
            notes=args.notes,
            config=args,
            resume="allow",
        )
        if wandb.run.resumed:
            print(
                "resuming wandb logging at step "
                + str(wandb.run.step)
                + " with run id "
                + str(wandb.run.id)
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

        # Uncomment for detailed logging (maybe make hyperparameter?)
        """wandb.watch(
            trainer.feature_extractor.features_model,
            log="all",
            log_freq=model_stats_log_freq,
        )

        for i in range(args.num_dynamics):
            wandb.watch(
                trainer.dynamics_list[i].dynamics_net,
                log="all",
                log_freq=model_stats_log_freq,
            )"""

    if args.load and len(args.download_model_from) > 0:
        # TODO: Figure out how to continue logging when loading an artifact
        trainer.load_models()

    model_path = "./models/" + args.exp_name
    if args.model_save_freq >= 0:
        model_path = model_path + "/checkpoints"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    try:
        trainer.train(debugging=args.debugging)
        print("Model finished training.")
    except KeyboardInterrupt:
        print("Training interrupted.")
        trainer.save_models(final=True)
        if not args.debugging:
            run.finish()
