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
        # policy, feature_extractor, dynamics


        self.feature_extractor = self.feature_extractor_class(
            policy=self.policy,
            features_shared_with_policy=False,
            device=self.device,
            feature_dim=hyperparameter["feature_dim"],
            layernormalize=hyperparameter["layernorm"],
        )

        self.dynamics_class = Dynamics
        self.num_dynamics = num_dynamics

        # Create a list of dynamics models. Their disagreement is used for learning.
        self.dynamics_list = []
        for i in range(num_dynamics):
            self.dynamics_list.append(
                self.dynamics_class(
                    # auxiliary_task=self.feature_extractor,
                    hidden_dim=self.feature_extractor.hidden_dim,
                    ac_space=self.feature_extractor.ac_space,
                    ob_mean=self.feature_extractor.ob_mean,
                    ob_std=self.feature_extractor.ob_std,
                    feature_dim=hyperparameter["feature_dim"],
                    device=self.device,
                    scope="dynamics_{}".format(i),
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
            vlog_freq=hyperparameter["video_log_freq"],
            debugging=hyperparameter["debugging"],
            dynamics_list=self.dynamics_list,
            dyn_loss_weight=hyperparameter["dyn_loss_weight"],
            auxiliary_task=self.feature_extractor,
            # whether to use the variance or the prediction error for the reward
            use_disagreement=use_disagreement,
            backprop_through_reward=hyperparameter["backprop_through_reward"],
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

        print("observation space: " + str(self.ob_space))
        print("action space: " + str(self.ac_space))

        try:
            path_name = "./statistics/" + self.hyperparameter["env"]
            if self.hyperparameter["resize_obs"] > 0:
                path_name = path_name + "_" + str(self.hyperparameter["resize_obs"])
            print("loading environment statistics from " + path_name)

            self.ob_mean = np.load(path_name + "/ob_mean.npy")
            self.ob_std = np.load(path_name + "/ob_std.npy")
        except FileNotFoundError:
            print("No statistics file found. Creating new one.")
            path_name = path_name + "/"
            os.makedirs(os.path.dirname(path_name))
            self.ob_mean, self.ob_std = random_agent_ob_mean_std(env, nsteps=100)
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
            "observation stats: "
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

        for i in range(self.num_dynamics):
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
        for i in range(self.num_dynamics):
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
