import argparse

import torch.nn
import numpy as np

from typing import Callable
from typing_extensions import Literal
from dataclasses import dataclass, field
from nupic.embodied.disagreement.policies import Dynamics, CnnPolicy
from nupic.embodied.utils.parser_utils import DataClassArgumentParser
from nupic.embodied.disagreement import Trainer
from nupic.embodied.disagreement.agents import PpoOptimizer

from experiments import CONFIGS

import os

import numpy as np

import logging

from typing import Tuple, Optional

@dataclass
class LoggingArguments:
    project_name: str = "multitask"
    log_dir: str = None
    snapshot_mode: Literal["all", "last", "gap", "none"] = field(
        default="none",
        metadata={
            "help": "(str): Policy for which snapshots to keep (or make at"
                    "all). Can be either 'all' (all iterations will be saved), 'last'"
                    "(only the last iteration will be saved), 'gap' (every snapshot_gap"
                    "iterations are saved), or 'none' (do not save snapshots)."
        }
    ),
    snapshot_gap: int = field(
        default=1,
        metadata={
            "help": "Gap between snapshot iterations. Waits this number"
                    "of iterations before taking another snapshot."
        }
    ),

    def __post_init__(self):
        if self.log_dir is None:
            logging.warn(
                "log_dir is not defined, attempting to default "
                "to environment variable CHECKPOINT_DIR/multitask"
            )
            if "CHECKPOINT_DIR" not in os.environ:
                raise KeyError(
                    "Environment variable CHECKPOINT_DIR not found, required "
                    "when log_dir is not defined in experiment config"
                )
            else:
                checkpoint_dir = os.path.join(os.environ["CHECKPOINT_DIR"], "multitask")
                logging.info(f"Defining log_dir as {checkpoint_dir}")


@dataclass
class ExperimentArguments:
    seed: int = 1
    timesteps: int = 15000000
    cpus_per_worker: float = 0
    gpus_per_worker: float = 0.14

@dataclass
class TrainingArguments:
    discount: float = 0.99
    eval_episodes: int = 3
    num_buffer_transitions: int = 1e6
    evaluation_frequency: int = 2
    task_update_frequency: int = 1
    target_update_tau: float = 5e-3
    buffer_batch_size: int = 2560
    num_grad_steps_scale: float = 0.5
    policy_lr: float = 3.91e-4
    qf_lr: float = 3.91e-4
    reward_scale: float = 1.0

    def __post_init__(self):
        self.num_buffer_transitions = int(self.num_buffer_transitions)


@dataclass
class NetworkArguments:
    net_type: str = "Dendrite_MLP"
    dim_context: int = 50
    num_tasks: int = 10
    kw: bool = True
    hidden_sizes: Tuple = (2048, 2048)
    num_segments: int = 1
    kw_percent_on: float = 0.33
    context_percent_on: float = 1.0
    weight_sparsity: float = 0.0
    weight_init: str = "modified"
    dendrite_init: str = "modified"
    dendritic_layer_class: str = "one_segment"
    output_nonlinearity: Optional[str] = None
    preprocess_module_type: str = "relu"
    preprocess_output_dim: int = 64
    representation_module_type: Optional[str] = None
    representation_module_dims: Optional[str] = None
    policy_min_log_std: float = -20.0
    policy_max_log_std: float = 2.0
    distribution: str = "TanhNormal"

    def __post_init__(self):
        self.policy_min_std = np.exp(self.policy_min_log_std)
        self.policy_max_std = np.exp(self.policy_max_log_std)


def create_exp_parser():
    return DataClassArgumentParser([
        LoggingArguments,
        ExperimentArguments,
        TrainingArguments,
        NetworkArguments,
    ])


def create_cmd_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        add_help=False,
    )
    # Run options
    parser.add_argument(
        "-e",
        "--exp_name",
        help="Experiment to run",
        choices=list(CONFIGS.keys())
    )
    parser.add_argument(
        "-n",
        "--wandb_run_name",
        default="",
        help="Name of the wandb run"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--local_only",
        action="store_true",
        default=False,
        help="Whether or not to log results to wandb"
    )
    parser.add_argument(
        "-c",
        "--cpu",
        action="store_true",
        default=False,
        help="Whether to use CPU even if GPU is available",
    )
    # TODO: evaluate whether or not debugging flag is required
    parser.add_argument(
        "-d",
        "--debugging",
        action="store_true",
        default=False,
        help="Option to use when debugging so not every test run is logged.",
    )

    return parser
