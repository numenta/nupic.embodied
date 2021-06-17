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

from functools import partial
import torch
import numpy as np
import gym
import os

from baselines.bench import Monitor
import wandb

from nupic.embodied.policies.auxiliary_tasks import (
    FeatureExtractor,
    InverseDynamics,
    VAE,
)
from nupic.embodied.policies.dynamics import Dynamics
from nupic.embodied.policies.curious_cnn_policy import CnnPolicy
from nupic.embodied.agents.curious_ppo_agent import PpoOptimizer
from nupic.embodied.utils.utils import random_agent_ob_mean_std


class Trainer(object):
    def __init__(
        self,
        make_env,
        hyperparameter,
        num_timesteps,
        envs_per_process,
        num_dynamics,
        use_disagreement,
        device,
    ):
        self.make_env = make_env
        self.hyperparameter = hyperparameter
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self.device = device
        self.wandb_id = None
        self.wandb_run = None
        self._set_env_vars()

        # Initialize the PPO policy for action selection
        self.policy = CnnPolicy(
            scope="policy",
            device=self.device,
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            feature_dim=hyperparameter["feature_dim"],
            hidden_dim=hyperparameter["policy_hidden_dim"],
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nonlinear=torch.nn.LeakyReLU,
        )

        """
        Select the feature extractor for the disagreement part.
        There are 4 options for the feature extraction:
        -----
        none: small conv net with no auxiliary task -> random features
        idf: Inverse dynamics (like curiosity module) with auxiliary task to predict
             action a_t from state s_t and s_t+1
        vaesph: small conv net + small deconv net with auxiliary task of variational
                autoencoder. The features are the hideen layer between encoder and
                decoder. Additionally a scaling factor is applied..
        vaenonsph: Same as vaesph but without the scaling factor.
        """
        self.feature_extractor_class = {
            "none": FeatureExtractor,
            "idf": InverseDynamics,
            "vaesph": partial(VAE, spherical_obs=True),
            "vaenonsph": partial(VAE, spherical_obs=False),
        }[hyperparameter["feat_learning"]]

        # Initialize the feature extractor
        self.feature_extractor = self.feature_extractor_class(
            policy=self.policy,
            features_shared_with_policy=False,
            device=self.device,
            feature_dim=hyperparameter["feature_dim"],
            layernormalize=hyperparameter["layernorm"],
        )

        self.dynamics_class = Dynamics

        # Create a list of dynamics models. Their disagreement is used for learning.
        self.dynamics_list = []
        for i in range(num_dynamics):
            self.dynamics_list.append(
                self.dynamics_class(
                    auxiliary_task=self.feature_extractor,
                    feature_dim=hyperparameter["feature_dim"],
                    device=self.device,
                    scope="dynamics_{}".format(i),
                    # whether to use the variance or the prediction error for the reward
                    use_disagreement=use_disagreement,
                )
            )

        # Initialize the agent.
        self.agent = PpoOptimizer(
            scope="ppo",
            device=self.device,
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            policy=self.policy,  # change to policy
            use_done=hyperparameter["use_done"],  # don't use the done information
            gamma=hyperparameter["gamma"],  # discount factor
            lam=hyperparameter["lambda"],  # discount factor for advantage
            nepochs=hyperparameter["nepochs"],
            nminibatches=hyperparameter["nminibatches"],
            lr=hyperparameter["lr"],
            cliprange=0.1,  # clipping policy gradient
            nsteps_per_seg=hyperparameter[
                "nsteps_per_seg"
            ],  # number of steps in each environment before taking a learning step
            nsegs_per_env=hyperparameter[
                "nsegs_per_env"
            ],  # how often to repeat before doing an update, 1
            entropy_coef=hyperparameter["entropy_coef"],  # entropy
            normrew=hyperparameter["norm_rew"],  # whether to normalize reward
            normadv=hyperparameter["norm_adv"],  # whether to normalize advantage
            ext_coeff=hyperparameter["ext_coeff"],  # weight of the environment reward
            int_coeff=hyperparameter["int_coeff"],  # weight of the disagreement reward
            exp_name=hyperparameter["exp_name"],
            vlog_freq=hyperparameter["video_log_freq"],
            debugging=hyperparameter["debugging"],
            dynamics_list=self.dynamics_list,
            backprop_through_reward=hyperparameter["backprop_through_reward"]
        )

        self.agent.start_interaction(
            self.envs,
            nlump=hyperparameter["nlumps"],
            dynamics_list=self.dynamics_list,
        )

    def _set_env_vars(self):
        """Set environment variables.

        Checks whether average observations have already been calculated (currently just
        for robt arm environment) and if yes loads it. Otherwise it calculates these
        statistics with a random agent.
        Currently nsteps is set to 100 to save time during debugging. Set this higher
        for experiments and save them to save time.

        """
        print("Create environment to collect statistics.")
        env = self.make_env(0)
        self.ob_space, self.ac_space = env.observation_space, env.action_space

        try:
            self.ob_mean = np.load(
                "./statistics/" + self.hyperparameter["env"] + "/ob_mean.npy"
            )
            self.ob_std = np.load(
                "./statistics/" + self.hyperparameter["env"] + "/ob_std.npy"
            )
            print("loaded environment statistics")
        except FileNotFoundError:
            print("No statistics file found. Creating new one.")
            path_name = "./statistics/" + self.hyperparameter["env"] + "/"
            os.makedirs(os.path.dirname(path_name))
            self.ob_mean, self.ob_std = random_agent_ob_mean_std(env, nsteps=10000)
            np.save(
                path_name + "/ob_mean.npy",
                self.ob_mean,
            )
            np.save(
                path_name + "/ob_std.npy",
                self.ob_std,
            )
            print("Saved environment statistics.")
        print(
            "obervation stats: "
            + str(np.mean(self.ob_mean))
            + " std: "
            + str(self.ob_std)
        )

        del env
        print("done.")
        self.envs = [partial(self.make_env, i) for i in range(self.envs_per_process)]

    def save_models(self, final=False):
        state_dicts = {
            "feature_extractor": self.feature_extractor.features_model.state_dict(),
            "policy_features": self.policy.features_model.state_dict(),
            "policy_hidden": self.policy.pd_hidden.state_dict(),
            "policy_pd_head": self.policy.pd_head.state_dict(),
            "policy_vf_head": self.policy.vf_head.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "step_count": self.agent.step_count,
            "n_updates": self.agent.n_updates,
            "total_secs": self.agent.total_secs,
            "best_ext_ret": self.agent.rollout.best_ext_return,
        }
        if self.hyperparameter["feat_learning"] == "idf":
            state_dicts["idf_fc"] = self.feature_extractor.fc.state_dict()
        elif self.hyperparameter["feat_learning"] == "vaesph":
            state_dicts[
                "decoder_model"
            ] = self.feature_extractor.decoder_model.state_dict()
            state_dicts["scale"] = self.feature_extractor.scale
        elif self.hyperparameter["feat_learning"] == "vaenonsph":
            state_dicts[
                "decoder_model"
            ] = self.feature_extractor.decoder_model.state_dict()

        if self.hyperparameter["norm_rew"]:
            state_dicts["tracked_reward"] = self.agent.reward_forward_filter.rewems
            state_dicts["reward_stats_mean"] = self.agent.reward_stats.mean
            state_dicts["reward_stats_var"] = self.agent.reward_stats.var
            state_dicts["reward_stats_count"] = self.agent.reward_stats.count

        if not self.hyperparameter["debugging"]:
            state_dicts["wandb_id"] = wandb.run.id

        for i in range(args.num_dynamics):
            state_dicts["dynamics_model_" + str(i)] = self.dynamics_list[
                i
            ].dynamics_net.state_dict()

        if final:
            model_path = "./models/" + self.hyperparameter["exp_name"]
            torch.save(state_dicts, model_path + "/model.pt")
            print("Saved final model at " + model_path + "/model.pt")
            if not self.hyperparameter["debugging"]:
                artifact = wandb.Artifact(self.hyperparameter["exp_name"], type="model")
                artifact.add_file(model_path + "/model.pt")
                self.wandb_run.log_artifact(artifact)
                # self.wandb_run.join()
                print("Model saved as artifact to wandb. Wait until sync is finished.")
        else:
            model_path = "./models/" + self.hyperparameter["exp_name"] + "/checkpoints"
            torch.save(
                state_dicts, model_path + "/model" + str(self.agent.step_count) + ".pt"
            )
            print(
                "Saved intermediate model at "
                + model_path
                + "/model"
                + str(self.agent.step_count)
                + ".pt"
            )

    def load_models(self):
        if self.hyperparameter["download_model_from"] != "":
            artifact = self.wandb_run.use_artifact(
                self.hyperparameter["download_model_from"], type="model"
            )
            model_path = artifact.download()
        else:
            model_path = "./models/" + self.hyperparameter["exp_name"]
        print("Loading model from " + str(model_path + "/model.pt"))
        checkpoint = torch.load(model_path + "/model.pt")

        self.feature_extractor.features_model.load_state_dict(
            checkpoint["feature_extractor"]
        )
        self.policy.features_model.load_state_dict(checkpoint["policy_features"])
        self.policy.pd_hidden.load_state_dict(checkpoint["policy_hidden"])
        self.policy.pd_head.load_state_dict(checkpoint["policy_pd_head"])
        self.policy.vf_head.load_state_dict(checkpoint["policy_vf_head"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])
        for i in range(args.num_dynamics):
            self.dynamics_list[i].dynamics_net.load_state_dict(
                checkpoint["dynamics_model_" + str(i)]
            )
        print("starting at step " + str(checkpoint["step_count"]))
        self.agent.start_step = checkpoint["step_count"]
        self.agent.n_updates = checkpoint["n_updates"]
        self.agent.time_trained_so_far = checkpoint["total_secs"]
        self.agent.rollout.best_ext_return = checkpoint["best_ext_ret"]

        if self.hyperparameter["feat_learning"] == "idf":
            self.feature_extractor.fc.load_state_dict(checkpoint["idf_fc"])
        elif self.hyperparameter["feat_learning"] == "vaesph":
            self.feature_extractor.decoder_model.load_state_dict(
                checkpoint["decoder_model"]
            )
            self.feature_extractor.scale = checkpoint["scale"]
        elif self.hyperparameter["feat_learning"] == "vaenonsph":
            self.feature_extractor.decoder_model.load_state_dict(
                checkpoint["decoder_model"]
            )

        if self.hyperparameter["norm_rew"]:
            self.agent.reward_forward_filter.rewems = checkpoint["tracked_reward"]
            self.agent.reward_stats.mean = checkpoint["reward_stats_mean"]
            self.agent.reward_stats.var = checkpoint["reward_stats_var"]
            self.agent.reward_stats.count = checkpoint["reward_stats_count"]

        if not self.hyperparameter["debugging"]:
            self.wandb_id = checkpoint["wandb_id"]
        print("Model successfully loaded.")

    def train(self):
        """Training loop for the agent.

        Keeps learning until num_timesteps is reached.

        """

        print("# of timesteps: " + str(self.num_timesteps))
        while True:
            info = self.agent.step()
            print("------------------------------------------")
            print(
                "Step count: "
                + str(self.agent.step_count)
                + " # Updates: "
                + str(self.agent.n_updates)
            )
            print("------------------------------------------")
            for i in info["update"]:
                try:
                    print(str(np.round(info["update"][i], 3)) + " - " + i)
                except TypeError:  # skip wandb elements (Histogram and Video)
                    pass
            if not args.debugging:
                wandb.log(info["update"])

            # Save intermediate model at specified save frequency
            if (
                self.hyperparameter["model_save_freq"] >= 0
                and self.agent.n_updates % self.hyperparameter["model_save_freq"] == 0
            ):
                print(
                    str(self.agent.n_updates) + " updates - saving intermediate model."
                )
                self.save_models()

            # Check if num_timesteps have been executed and end run.
            if self.agent.step_count >= self.num_timesteps:
                print("step count > num timesteps - " + str(self.num_timesteps))
                self.save_models(final=True)
                break
        print("Stopped interaction")
        self.agent.stop_interaction()


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
        from nupic.embodied.envs.wrappers import (
            MontezumaInfoWrapper,
            MaxAndSkipEnv,
            ProcessFrame84,
            ExtraTimeLimit,
            StickyActionEnv,
            AddRandomStateToInfo
        )
        from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
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

    print("adding monitor")
    env = Monitor(env, filename=None)
    return env


if __name__ == "__main__":
    import argparse

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
        trainer.train()
        print("Model finished training.")
    except KeyboardInterrupt:
        print("Training interrupted.")
        trainer.save_models(final=True)
        if not args.debugging:
            run.finish()
