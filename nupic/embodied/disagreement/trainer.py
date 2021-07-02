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
#
import os
from functools import partial

import numpy as np
import torch
import wandb

from nupic.embodied.disagreement.agents import PpoOptimizer
from nupic.embodied.disagreement.policies import (
    VAE,
    FeatureExtractor,
    InverseDynamics,
)
from nupic.embodied.utils.misc import random_agent_ob_mean_std

FEATURE_EXTRACTOR_CLASS_MAPPING = {
    "none": FeatureExtractor,
    "idf": InverseDynamics,
    "vaesph": partial(VAE, spherical_obs=True),
    "vaenonsph": partial(VAE, spherical_obs=False),
}

class Trainer(object):
    def __init__(
        self,
        exp_name,
        make_env,
        device,
        logging_args,
        env_args,
        trainer_args,
        learner_args,
        debugging=False
    ):
        self.exp_name = exp_name
        self.wandb_id = None
        self.wandb_run = None
        self.feat_learning = trainer_args.feat_learning
        self.num_timesteps = trainer_args.num_timesteps
        self.model_save_freq = logging_args.model_save_freq
        envs, ob_space, ac_space, ob_mean, ob_std = self.init_environments(
            make_env, env_args
        )

        # Initialize the PPO policy for action selection
        self.policy = trainer_args.policy_class(
            scope="policy",
            device=device,
            ob_space=ob_space,
            ac_space=ac_space,
            feature_dim=trainer_args.feature_dim,
            hidden_dim=trainer_args.policy_hidden_dim,
            ob_mean=ob_mean,
            ob_std=ob_std,
            layernormalize=trainer_args.policy_layernorm,
            nonlinear=trainer_args.policy_nonlinearity,
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
        self.feature_extractor_class = \
            FEATURE_EXTRACTOR_CLASS_MAPPING[trainer_args.feat_learning]

        self.feature_extractor = self.feature_extractor_class(
            policy=self.policy,
            device=device,
            layernormalize=trainer_args.fextractor_layernorm,
            feature_dim=trainer_args.feature_dim,
            features_shared_with_policy=trainer_args.features_shared_with_policy
        )

        self.num_dynamics = trainer_args.num_dynamics

        # Create a list of dynamics models. Their disagreement is used for learning.
        self.dynamics_list = []
        for i in range(self.num_dynamics):
            self.dynamics_list.append(
                trainer_args.dynamics_class(
                    hidden_dim=trainer_args.policy_hidden_dim,
                    ac_space=ac_space,
                    ob_mean=ob_mean,
                    ob_std=ob_std,
                    feature_dim=trainer_args.feature_dim,
                    device=device,
                    scope="dynamics_{}".format(i),
                )
            )

        # Initialize the agent.
        self.agent = PpoOptimizer(
            scope="ppo",
            device=device,
            ob_space=ob_space,
            ac_space=ac_space,
            policy=self.policy,
            vlog_freq=logging_args.video_log_freq,
            debugging=debugging,
            dynamics_list=self.dynamics_list,
            auxiliary_task=self.feature_extractor,
            **learner_args.__dict__
        )

        self.agent.start_interaction(envs, nlump=trainer_args.nlumps)

    def init_environments(self, make_env, env_args):
        """Set environment variables.

        Checks whether average observations have already been calculated (currently just
        for robot arm environment) and if yes loads it. Otherwise it calculates these
        statistics with a random agent.
        Currently nsteps is set to 100 to save time during debugging. Set this higher
        for experiments and save them to save time.

        """
        print("Create environment to collect statistics.")
        sample_env = make_env(0)
        ob_space, ac_space = sample_env.observation_space, sample_env.action_space

        print("observation space: " + str(ob_space))
        print("action space: " + str(ac_space))

        try:
            path_name = "./statistics/" + env_args.env
            if env_args.resize_obs > 0:
                path_name = path_name + "_" + str(env_args.resize_obs)
            print("loading environment statistics from " + path_name)

            ob_mean = np.load(path_name + "/ob_mean.npy")
            ob_std = np.load(path_name + "/ob_std.npy")
        except FileNotFoundError:
            print("No statistics file found. Creating new one.")
            path_name = path_name + "/"
            os.makedirs(os.path.dirname(path_name))
            ob_mean, ob_std = random_agent_ob_mean_std(
                sample_env, nsteps=env_args.nsteps_to_collect_statistics
            )
            np.save(
                path_name + "/ob_mean.npy",
                ob_mean,
            )
            np.save(
                path_name + "/ob_std.npy",
                ob_std,
            )
            print("Saved environment statistics.")
        print(
            "observation stats: "
            + str(np.mean(ob_mean))
            + " std: "
            + str(ob_std)
        )

        del sample_env
        print("done.")
        envs = [partial(make_env, i) for i in range(env_args.envs_per_process)]

        return envs, ob_space, ac_space, ob_mean, ob_std

    def save_models(self, debugging=False, final=False, model_dir=None):

        state_dicts = sds = {
            "feature_extractor": self.feature_extractor.features_model.state_dict(),
            "policy_features": self.policy.features_model.state_dict(),
            "policy_hidden": self.policy.pd_hidden.state_dict(),
            "policy_pd_head": self.policy.pd_head.state_dict(),
            "policy_vf_head": self.policy.vf_head.state_dict(),
            "step_count": self.agent.step_count,
            "n_updates": self.agent.n_updates,
            "total_secs": self.agent.total_secs,
            "best_ext_ret": self.agent.rollout.best_ext_return,
        }

        if self.agent.backprop_through_reward:
            sds["policy_optimizer"] = self.agent.policy_optimizer.state_dict()
            sds["dynamics_optimizer"] = self.agent.dynamics_optimizer.state_dict()
        else:
            sds["optimizer"] = self.agent.optimizer.state_dict()

        if self.feat_learning == "idf":
            sds["idf_fc"] = self.feature_extractor.fc.state_dict()
        elif self.feat_learning == "vaesph":
            sds["decoder_model"] = self.feature_extractor.decoder_model.state_dict()
            sds["scale"] = self.feature_extractor.scale
        elif self.feat_learning == "vaenonsph":
            sds["decoder_model"] = self.feature_extractor.decoder_model.state_dict()

        if self.agent.norm_rew:
            sds["tracked_reward"] = self.agent.reward_forward_filter.rewems
            sds["reward_stats_mean"] = self.agent.reward_stats.mean
            sds["reward_stats_var"] = self.agent.reward_stats.var
            sds["reward_stats_count"] = self.agent.reward_stats.count

        if not debugging:
            sds["wandb_id"] = wandb.run.id

        for i in range(self.num_dynamics):
            sds[f"dynamics_model_{i}"] = self.dynamics_list[i].dynamics_net.state_dict()

        if final:
            model_path = os.path.join(model_dir, "model.pt")
            torch.save(state_dicts, model_path)
            print(f"Saved final model at {model_path}")
            if not debugging:
                artifact = wandb.Artifact(self.exp_name, type="model")
                artifact.add_file(model_path)
                self.wandb_run.log_artifact(artifact)
                # self.wandb_run.join()  # TODO: remove it?
                print("Model saved as artifact to wandb. Wait until sync is finished.")
        else:
            model_dir = os.path.join(model_dir, "checkpoints")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, f"model{self.agent.step_count}.pt")

            torch.save(state_dicts, model_path)
            print("Saved intermediate model at {model_path}")

    def load_models(self, debugging=False, download_model_from=None, model_dir=None):
        """
        Load existing model to continue training

        TODO: Figure out how to continue logging when loading an artifact
        TODO: If loading from checkpoint will fail - need to look for latest
        """

        if download_model_from is not None:
            artifact = self.wandb_run.use_artifact(download_model_from, type="model")
            model_path = artifact.download()
        else:
            model_path = os.path.join(model_dir, "model.pt")
        print(f"Loading model from {model_path}")
        checkpoint = cp = torch.load(model_path)  # noqa: F841

        self.feature_extractor.features_model.load_state_dict(cp["feature_extractor"])
        self.policy.features_model.load_state_dict(cp["policy_features"])
        self.policy.pd_hidden.load_state_dict(cp["policy_hidden"])
        self.policy.pd_head.load_state_dict(cp["policy_pd_head"])
        self.policy.vf_head.load_state_dict(cp["policy_vf_head"])

        if self.agent.backprop_through_reward:
            self.agent.policy_optimizer.load_state_dict(cp["policy_optimizer"])
            self.agent.dynamics_optimizer.load_state_dict(cp["dynamics_optimizer"])
        else:
            self.agent.optimizer.load_state_dict(cp["optimizer"])

        for i in range(self.num_dynamics):
            self.dynamics_list[i].dynamics_net.load_state_dict(
                cp[f"dynamics_model_{i}"]
            )
        print("starting at step " + str(cp["step_count"]))
        self.agent.start_step = cp["step_count"]
        self.agent.n_updates = cp["n_updates"]
        self.agent.time_trained_so_far = cp["total_secs"]
        self.agent.rollout.best_ext_return = cp["best_ext_ret"]

        if self.feat_learning == "idf":
            self.feature_extractor.fc.load_state_dict(cp["idf_fc"])
        elif self.feat_learning == "vaesph":
            self.feature_extractor.decoder_model.load_state_dict(cp["decoder_model"])
            self.feature_extractor.scale = cp["scale"]
        elif self.feat_learning == "vaenonsph":
            self.feature_extractor.decoder_model.load_state_dict(cp["decoder_model"])

        if self.save_params["norm_rew"]:
            self.agent.reward_forward_filter.rewems = cp["tracked_reward"]
            self.agent.reward_stats.mean = cp["reward_stats_mean"]
            self.agent.reward_stats.var = cp["reward_stats_var"]
            self.agent.reward_stats.count = cp["reward_stats_count"]

        if not debugging:
            self.wandb_id = cp["wandb_id"]
        print("Model successfully loaded.")

    def train(self, debugging=False):
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
            if not debugging:
                wandb.log(info["update"])

            # Save intermediate model at specified save frequency
            if (
                self.model_save_freq >= 0
                and self.agent.n_updates % self.model_save_freq == 0
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
