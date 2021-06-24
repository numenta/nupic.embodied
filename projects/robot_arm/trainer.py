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

from nupic.embodied.agents.curious_ppo_agent import PpoOptimizer
from nupic.embodied.policies.auxiliary_tasks import (
    VAE,
    FeatureExtractor,
    InverseDynamics,
)
from nupic.embodied.policies.curious_cnn_policy import CnnPolicy
from nupic.embodied.policies.dynamics import Dynamics
from nupic.embodied.utils.utils import random_agent_ob_mean_std


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
        self.wandb_id = None
        self.wandb_run = None
        self.exp_name = exp_name
        self.num_timesteps = trainer_args.num_timesteps
        self.model_save_freq = logging_args.model_save_freq
        envs, ob_space, ac_space, ob_mean, ob_std = self.init_environments(
            make_env, env_args
        )

        # Params required to save the model later
        self.save_params = dict(
            debugging=debugging,
            norm_rew=learner_args.norm_rew,
            feat_learning=trainer_args.feat_learning
        )

        # Initialize the PPO policy for action selection
        self.policy = CnnPolicy(
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
        self.feature_extractor_class = {
            "none": FeatureExtractor,
            "idf": InverseDynamics,
            "vaesph": partial(VAE, spherical_obs=True),
            "vaenonsph": partial(VAE, spherical_obs=False),
        }[trainer_args.feat_learning]

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
        if self.save_params["feat_learning"] == "idf":
            state_dicts["idf_fc"] = self.feature_extractor.fc.state_dict()
        elif self.save_params["feat_learning"] == "vaesph":
            state_dicts[
                "decoder_model"
            ] = self.feature_extractor.decoder_model.state_dict()
            state_dicts["scale"] = self.feature_extractor.scale
        elif self.save_params["feat_learning"] == "vaenonsph":
            state_dicts[
                "decoder_model"
            ] = self.feature_extractor.decoder_model.state_dict()

        if self.save_params["norm_rew"]:
            state_dicts["tracked_reward"] = self.agent.reward_forward_filter.rewems
            state_dicts["reward_stats_mean"] = self.agent.reward_stats.mean
            state_dicts["reward_stats_var"] = self.agent.reward_stats.var
            state_dicts["reward_stats_count"] = self.agent.reward_stats.count

        if not self.save_params["debugging"]:
            state_dicts["wandb_id"] = wandb.run.id

        for i in range(self.num_dynamics):
            state_dicts["dynamics_model_" + str(i)] = self.dynamics_list[
                i
            ].dynamics_net.state_dict()

        if final:
            model_path = "./models/" + self.exp_name
            torch.save(state_dicts, model_path + "/model.pt")
            print("Saved final model at " + model_path + "/model.pt")
            if not self.save_params["debugging"]:
                artifact = wandb.Artifact(self.exp_name, type="model")
                artifact.add_file(model_path + "/model.pt")
                self.wandb_run.log_artifact(artifact)
                # self.wandb_run.join()
                print("Model saved as artifact to wandb. Wait until sync is finished.")
        else:
            model_path = "./models/" + self.exp_name + "/checkpoints"
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

    def load_models(self, download_model_from=None):
        if download_model_from is not None:
            artifact = self.wandb_run.use_artifact(download_model_from, type="model")
            model_path = artifact.download()
        else:
            model_path = "./models/" + self.exp_name
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
        for i in range(self.num_dynamics):
            self.dynamics_list[i].dynamics_net.load_state_dict(
                checkpoint["dynamics_model_" + str(i)]
            )
        print("starting at step " + str(checkpoint["step_count"]))
        self.agent.start_step = checkpoint["step_count"]
        self.agent.n_updates = checkpoint["n_updates"]
        self.agent.time_trained_so_far = checkpoint["total_secs"]
        self.agent.rollout.best_ext_return = checkpoint["best_ext_ret"]

        if self.save_params["feat_learning"] == "idf":
            self.feature_extractor.fc.load_state_dict(checkpoint["idf_fc"])
        elif self.save_params["feat_learning"] == "vaesph":
            self.feature_extractor.decoder_model.load_state_dict(
                checkpoint["decoder_model"]
            )
            self.feature_extractor.scale = checkpoint["scale"]
        elif self.save_params["feat_learning"] == "vaenonsph":
            self.feature_extractor.decoder_model.load_state_dict(
                checkpoint["decoder_model"]
            )

        if self.save_params["norm_rew"]:
            self.agent.reward_forward_filter.rewems = checkpoint["tracked_reward"]
            self.agent.reward_stats.mean = checkpoint["reward_stats_mean"]
            self.agent.reward_stats.var = checkpoint["reward_stats_var"]
            self.agent.reward_stats.count = checkpoint["reward_stats_count"]

        if not self.save_params["debugging"]:
            self.wandb_id = checkpoint["wandb_id"]
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
