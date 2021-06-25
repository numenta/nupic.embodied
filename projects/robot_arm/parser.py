import argparse

import torch.nn

from typing import Callable, Literal
from dataclasses import dataclass, field
from nupic.embodied.policies.dynamics import Dynamics
from helper_functions import DataClassArgumentParser
from nupic.embodied.policies.curious_cnn_policy import CnnPolicy

from experiments import CONFIGS

@dataclass
class LoggingArguments:
    group: str = ""
    notes: str = ""
    video_log_freq: int = -1  # frequencies in num_updates (not time_steps)
    model_save_freq: int = -1
    detailed_wandb_logging: bool = False


@dataclass
class EnvironmentArguments:
    env: str = field(
        default="BreakoutNoFrameskip-v4",
        metadata={"help": "environment ID"}
    )
    env_kind: Literal["atari", "mario", "retro_multi", "roboarm"] = "atari"
    resize_obs: int = field(
        default=0,
        metadata={"help": "With current CNN possible sizes are "
                          "36, 44, 52, 60, 68, 76, 84, ..., 180"}
    )
    max_episode_steps: int = 4500
    noop_max: int = 30
    act_repeat: int = 10
    stickyAtari: bool = True
    crop_obs: bool = True
    touch_reward: bool = False
    random_force: bool = False
    envs_per_process: int = 128
    nsteps_to_collect_statistics: int = 10000

@dataclass
class TrainerArguments:
    nlumps: int = 1
    num_dynamics: int = 5
    num_timesteps: int = 1024
    feat_learning: Literal["none", "idf", "vaesph", "vaenonsph"] = "none"
    fextractor_layernorm: bool = True
    feature_dim: int = 512
    features_shared_with_policy: bool = False
    policy_layernorm: bool = False
    policy_hidden_dim: int = 512
    policy_class: Callable = CnnPolicy
    policy_nonlinearity: Callable = torch.nn.LeakyReLU
    dynamics_class: Callable = Dynamics


@dataclass
class LearnerArguments:
    use_done: int = 0
    ext_coeff: float = 0.0
    int_coeff: float = 1.0
    dyn_loss_weight: float = 1.0
    lam: float = 0.95
    gamma: float = 0.99
    norm_adv: int = 1
    norm_rew: int = 1
    lr: float = 1e-4
    entropy_coef: float = 0.001
    nepochs: int = 3
    nminibatches: int = 8
    cliprange: float = 0.1
    nsteps_per_seg: int = 128
    nsegs_per_env: int = 1
    backprop_through_reward: bool = False
    use_disagreement: bool = True


def create_exp_parser():
    return DataClassArgumentParser([
        LoggingArguments,
        EnvironmentArguments,
        TrainerArguments,
        LearnerArguments,
    ])


def create_cmd_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        add_help=False,
    )
    # Run options
    parser.add_argument(
        "-e", "--exp_name", help="Experiment to run",
        choices=list(CONFIGS.keys())
    )
    parser.add_argument(
        "-l", "--load", action="store_true", default=False,
        help="Specify --load to load an existing model (needs to have same exp_name)"
    )
    parser.add_argument(
        "-m", "--download_model_from", type=str, default="",
        help="Download a model from a wandb artifact (specify path)"
    )
    parser.add_argument(
        "-d", "--debugging", action="store_true", default=False,
        help="Option to use when debugging so not every test run is logged."
    )
    parser.add_argument(
        "-c", "--cpu", action="store_true", default=False,
        help="Whether to use CPU even if GPU is available"
    )
    return parser
