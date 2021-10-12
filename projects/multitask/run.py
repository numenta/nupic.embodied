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

import logging
import sys
import torch

from args_parser import create_cmd_parser, create_exp_parser
from experiments import CONFIGS

from nupic.embodied.multitask.trainer import Trainer
from nupic.embodied.utils.parser_utils import merge_args

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
    trainer_args = merge_args(exp_parser.parse_dict(CONFIGS[run_args.exp_name]))

    # Setup logging based on verbose defined by the user
    # TODO: logger level being overriden to WARN by garage, requires fixing
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if run_args.verbose else logging.INFO
    )
    logging.debug("Logger setup to debug mode")

    # Gives an additional option to define a wandb run name
    if run_args.wandb_run_name != "":
        run_args.wandb_run_name = run_args.exp_name

    # Automatically detects whether or not to use GPU
    use_gpu = torch.cuda.is_available() and not run_args.cpu
    print(f"Using GPU: {use_gpu}")
    print(trainer_args)

    trainer = Trainer(
        experiment_name=run_args.exp_name,
        use_gpu=use_gpu,
        trainer_args=trainer_args
    )

    if trainer_args.debug_mode:
        state_dict = trainer.state_dict()
    elif trainer_args.do_train:
        trainer.train(
            use_wandb=not run_args.local_only,
            evaluation_frequency=trainer_args.evaluation_frequency,
            checkpoint_frequency=trainer_args.checkpoint_frequency
        )
