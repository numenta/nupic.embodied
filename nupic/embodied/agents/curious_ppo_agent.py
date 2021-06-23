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
import time
from collections import Counter

import numpy as np
import torch
import wandb

from nupic.embodied.envs.rollout import Rollout
from nupic.embodied.envs.vec_env import ShmemVecEnv as VecEnv
from nupic.embodied.utils.model_parts import flatten_dims
from nupic.embodied.utils.mpi import mpi_moments
from nupic.embodied.utils.torch import convert_log_to_numpy, to_numpy
from nupic.embodied.utils.utils import (
    RunningMeanStd,
    explained_variance,
    get_mean_and_std,
)


class PpoOptimizer(object):
    """PPO optimizer used for learning from the rewards.

    Parameters
    ----------
    scope : str
        Scope name.
    device: torch.device
        Which device to optimize the model on.
    ob_space : Space
        Observation space properties (from env.observation_space).
    ac_space : Space
        Action space properties (from env.action_space).
    policy : CnnPolicy
        Stochastic policy to use for action selection.
    entropy_coef : float
        Weighting of entropy in the policy in the overall loss.
    gamma : float
        Discount factor for rewards over time.
    lam : float
        Discount factor lambda for calculating advantages.
    nepochs : int
        Number of epochs for updates of the network parameters.
    lr : float
        Learnig rate of the optimizer.
    cliprange : float
        PPO clipping parameter.
    nminibatches : int
        Number of minibatches.
    normrew : bool
        Whether to apply the RewardForwardFilter and normalize rewards.
    normadv : bool
        Whether to normalize the advantages.
    use_done : bool
        Whether to take into account new episode (done=True) in the advantage
        calculation.
    ext_coeff : float
        Weighting of the external rewards in the overall rewards.
    int_coeff : float
        Weighting of the internal rewards (disagreement) in the overall rewards.
    nsteps_per_seg : int
        Number of environment steps per update segment.
    nsegs_per_env : int
        Number of segments to collect in each environment.
    vlog_freq : int
        After how many steps should a video of the training be logged.
    dynamics_list : [Dynamics]
        List of dynamics models to use for internal reward calculation.
    use_disagreement : bool
        If loss should be calculated from the variance over dynamics model features.
        If false, the loss is the variance over the error of state features and next
        state features between the different dynamics models.
    dyn_loss_weight: float
        Weighting of the dynamic loss in the total loss.

    Attributes
    ----------
    n_updates : int
        Number of updates that were performed so far.
    envs : [VecEnv]
        List of vector enviornments to use for experience collection.

    """

    envs = None

    def __init__(
        self,
        scope,
        device,
        ob_space,
        ac_space,
        policy,
        entropy_coef,
        gamma,
        lam,
        nepochs,
        lr,
        cliprange,
        nminibatches,
        normrew,
        normadv,
        use_done,
        ext_coeff,
        int_coeff,
        nsteps_per_seg,
        nsegs_per_env,
        vlog_freq,
        debugging,
        dynamics_list,
        dyn_loss_weight,
        auxiliary_task,
        use_disagreement,
        backprop_through_reward=False,
    ):
        self.dynamics_list = dynamics_list
        self.n_updates = 0
        self.scope = scope
        self.device = device
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.policy = policy
        self.nepochs = nepochs
        self.lr = lr
        self.cliprange = cliprange
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nminibatches = nminibatches
        self.gamma = gamma
        self.lam = lam
        self.normrew = normrew
        self.normadv = normadv
        self.use_done = use_done
        self.entropy_coef = entropy_coef
        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff
        self.vlog_freq = vlog_freq
        self.debugging = debugging
        self.time_trained_so_far = 0
        self.dyn_loss_weight = dyn_loss_weight
        self.auxiliary_task = auxiliary_task
        self.use_disagreement = use_disagreement
        self.backprop_through_reward = backprop_through_reward

    def start_interaction(self, env_fns, nlump=1):
        """Set up environments and initialize everything.

        Parameters
        ----------
        env_fns : [envs]
            List of environments (functions), optionally with wrappers etc.
        nlump : int
            ..

        """
        # Specify parameters that should be optimized
        # auxiliary task params is the same for all dynamic models
        policy_param_list = [*self.policy.param_list]
        dynamics_param_list = [*self.auxiliary_task.param_list]
        for dynamic in self.dynamics_list:
            dynamics_param_list.extend(dynamic.param_list)

        # Initialize the optimizer
        if self.backprop_through_reward:
            self.policy_optimizer = torch.optim.Adam(policy_param_list, lr=self.lr)
            self.dynamics_optimizer = torch.optim.Adam(dynamics_param_list, lr=self.lr)
        else:
            params_list = policy_param_list + dynamics_param_list
            self.optimizer = torch.optim.Adam(params_list, lr=self.lr)

        # set parameters
        self.nenvs = nenvs = len(env_fns)
        self.nlump = nlump
        self.lump_stride = nenvs // self.nlump
        # Initialize list of VecEnvs
        self.envs = [
            VecEnv(
                env_fns[lump * self.lump_stride : (lump + 1) * self.lump_stride],
                spaces=[self.ob_space, self.ac_space],
            )
            for lump in range(self.nlump)
        ]

        # Initialize rollout class for experience collection
        self.rollout = Rollout(
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            nenvs=nenvs,
            nsteps_per_seg=self.nsteps_per_seg,
            nsegs_per_env=self.nsegs_per_env,
            nlumps=self.nlump,
            envs=self.envs,
            policy=self.policy,
            int_rew_coeff=self.int_coeff,
            ext_rew_coeff=self.ext_coeff,
            dynamics_list=self.dynamics_list,
        )

        def empty_tensor(shape):
            return torch.zeros(shape, dtype=torch.float32, device=self.device)

        # Initialize replay buffers for advantages and returns of each rollout
        self.buf_advantages = empty_tensor((nenvs, self.rollout.nsteps))
        self.buf_returns = empty_tensor((nenvs, self.rollout.nsteps))

        if self.normrew:  # if normalize reward, defaults to True
            # Sum up and discount rewards
            self.reward_forward_filter = RewardForwardFilter(self.gamma)
            # Initialize running mean and std tracker
            self.reward_stats = RunningMeanStd()

        self.step_count = 0
        self.start_step = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

        # Set internal step loop parameters
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        self.envs_per_batch = max(1, envsperbatch)

    def stop_interaction(self):
        """Close environments when stopping."""
        for env in self.envs:
            env.close()

    def calculate_advantages(self, rews, use_done, gamma, lam):
        """Calculate advantages from the rewards.

        Parameters
        ----------
        rews : array
            rewards. shape = [n_envs, n_steps]
        use_done : bool
            Whether to use news (which are the done infos from the environment).
        gamma : float
            Discount factor for the rewards.
        lam : float
            Generalized advantage estimator smoothing parameter.

        """
        nsteps = self.rollout.nsteps
        last_gae_lambda = 0
        for t in range(nsteps - 1, -1, -1):  # going backwards from last step in seg
            # Is the next step a new run (done=True)?
            nextnew = (
                self.rollout.buf_dones[:, t + 1]
                if t + 1 < nsteps
                else self.rollout.buf_done_last
            )
            if not use_done:
                nextnew = 0
            nextnotnew = 1 - nextnew

            # The value estimate of the next time step
            nextvals = (
                self.rollout.buf_vpreds[:, t + 1]
                if t + 1 < nsteps
                else self.rollout.buf_vpred_last
            )
            # difference between current reward + discounted value estimate of the next
            # state and the current value estimate -> TD error
            delta = (
                rews[:, t]
                + gamma * nextvals * nextnotnew
                - self.rollout.buf_vpreds[:, t]
            )
            # Calculate advantages and put in the buffer (delta + discounted last
            # advantage)
            self.buf_advantages[:, t] = last_gae_lambda = (
                delta + gamma * lam * nextnotnew * last_gae_lambda
            )
        # Update return buffer (advantages + value estimates)
        self.buf_returns[:] = self.buf_advantages + self.rollout.buf_vpreds

    def log_pre_update(self):
        """
        Initialize the info dictionary to be logged in wandb and collect base metrics
        Returns info dictionary.
        """

        # Initialize and update the info dict for logging
        info = dict()
        info["ppo/advantage_mean"] = self.buf_advantages.mean()
        info["ppo/advantage_std"] = self.buf_advantages.std()
        info["ppo/return_mean"] = self.buf_returns.mean()
        info["ppo/return_std"] = self.buf_returns.std()
        info["ppo/value_est_mean"] = self.rollout.buf_vpreds.mean()
        info["ppo/value_est_std"] = self.rollout.buf_vpreds.std()
        info["ppo/explained_variance"] = explained_variance(
            self.rollout.buf_vpreds.flatten(),  # TODO: switch to ravel if pytorch>=1.9
            self.buf_returns.flatten()  # TODO: switch to ravel if pytorch >= 1.9
        )
        info["ppo/reward_mean"] = torch.mean(self.rollout.buf_rewards)

        if self.rollout.best_ext_return is not None:
            info["performance/best_ext_return"] = self.rollout.best_ext_return
        # TODO: maybe add extra flag for detailed logging so runs are not slowed down
        if not self.debugging:
            feature_stats, stacked_act_feat = self.get_activation_stats(
                self.rollout.buf_acts_features, "activations_features/"
            )
            hidden_stats, stacked_act_pi = self.get_activation_stats(
                self.rollout.buf_acts_pi, "activations_hidden/"
            )
            info.update(feature_stats)
            info.update(hidden_stats)

            info["activations_features/raw_act_distribution"] = wandb.Histogram(
                to_numpy(stacked_act_feat)
            )
            info["activations_hidden/raw_act_distribution"] = wandb.Histogram(
                to_numpy(stacked_act_pi)
            )

            info["ppo/action_distribution"] = wandb.Histogram(
                to_numpy(self.rollout.buf_acs).flatten()
            )

            if self.vlog_freq >= 0 and self.n_updates % self.vlog_freq == 0:
                print(str(self.n_updates) + " updates - logging video.")
                # Reshape images such that they have shape [time,channels,width,height]
                sample_video = torch.moveaxis(self.rollout.buf_obs[0], 3, 1)
                # Log buffer video from first env
                info["observations"] = wandb.Video(
                    to_numpy(sample_video), fps=12, format="gif"
                )

        return info

    def collect_rewards(self, normalize=True):
        """Outputs a torch Tensor"""
        if normalize:  # default=True
            discounted_rewards = [
                self.reward_forward_filter.update(rew)
                for rew in self.rollout.buf_rewards.T
            ]
            discounted_rewards = torch.stack(discounted_rewards)
            # rewards_mean, rewards_std, rewards_count
            rewards_mean, rewards_std, rewards_count = mpi_moments(
                discounted_rewards.flatten()  # TODO: switch to ravel in pytorch >= 1.9
            )
            # reward forward filter running mean std
            self.reward_stats.update_from_moments(
                rewards_mean, rewards_std ** 2, rewards_count
            )
            return self.rollout.buf_rewards / torch.sqrt(self.reward_stats.var)
        else:
            # Copy directly from buff rewards
            # TODO: should it be detached? is gradient propagating through buf rewards?
            return self.rollout.buf_rewards.clone().detach()

    def load_returns(self, idxs):
        return self.buf_advantages[idxs], self.buf_returns[idxs]

    def learn(self):
        """Calculate losses and update parameters based on current rollout.

        Returns
        -------
        info
            Dictionary of infos about the current update and training statistics.

        """
        print("Calculate Statistics")
        rews = self.collect_rewards(normalize=self.normrew)

        # Calculate advantages using the current rewards and value estimates
        self.calculate_advantages(
            rews=rews, use_done=self.use_done, gamma=self.gamma, lam=self.lam
        )

        info = self.log_pre_update()
        to_report = Counter()

        # TODO: we are logging buf_advantages before normalizing, is that correct?
        if self.normadv:  # defaults to True
            # normalize advantages
            m, s = get_mean_and_std(self.buf_advantages)
            self.buf_advantages = (self.buf_advantages - m) / (s + 1e-7)

        # Update the networks & get losses for nepochs * nminibatches
        print("Update the models")
        for idx in range(self.nepochs):
            print(f"Starting epoch: {idx+1}/{self.nepochs}")
            env_idxs = np.random.permutation(self.nenvs * self.nsegs_per_env)
            total_segs = self.nenvs * self.nsegs_per_env
            for idx, start in enumerate(range(0, total_segs, self.envs_per_batch)):
                print(f"Starting batch: {idx+1}/{total_segs//self.envs_per_batch}")
                minibatch_idxs = env_idxs[start:start + self.envs_per_batch]

                # Get rollout experiences for current minibatch
                acs, rews, neglogprobs, obs, last_obs = self.rollout.load_from_buffer(
                    minibatch_idxs
                )
                advantages, returns = self.load_returns(minibatch_idxs)

                if self.backprop_through_reward:
                    loss_info = self.update_step_backprop(acs, obs, last_obs)
                else:
                    loss_info = self.update_step_ppo(
                        acs, obs, last_obs, neglogprobs, advantages, returns
                    )

                # Update counter with info gathered from aux loss and loss
                for metric, value in loss_info.items():
                    to_report[metric] += value

        num_steps_taken = self.nminibatches * self.nepochs
        to_report = {k: v / num_steps_taken for k, v in to_report.items()}
        info.update(to_report)
        self.n_updates += 1

        info = self.log_post_update(info)

        return info

    def update_step_ppo(
        self, acs, obs, last_obs, neglogprobs, advantages, returns
    ):
        """Regular update step in exploration by disagreement using PPO"""

        acs, features, next_features = self.update_auxiliary_task(
            acs, obs, last_obs, return_next_features=True
        )

        self.optimizer.zero_grad()
        # TODO: aux task could be skipped if not using aux task
        aux_loss, aux_loss_info = self.auxiliary_loss()
        dyn_loss, dyn_loss_info = self.dynamics_loss(acs, features, next_features)
        policy_loss, loss_info = self.ppo_loss(
            acs, neglogprobs, advantages, returns
        )  # forward
        total_loss = aux_loss + dyn_loss + policy_loss
        total_loss.backward()
        self.optimizer.step()

        loss_info.update(aux_loss_info)
        loss_info.update(dyn_loss_info)
        loss_info["loss/total_loss"] = total_loss

        return loss_info

    def update_step_backprop(self, acs, obs, last_obs):
        """
        Update when using backpropagation through rewards instead of PPO
        Alternate between steps to the dynamics loss and the policy loss
        TODO: do we need two update auxiliary tasks in this 2-stage training loop?
        """

        acs, features, next_features = self.update_auxiliary_task(
            acs, obs, last_obs, return_next_features=True
        )

        self.dynamics_optimizer.zero_grad()
        aux_loss, aux_loss_info = self.auxiliary_loss()
        dyn_loss, dyn_loss_info = self.dynamics_loss(acs, features, next_features)
        total_dyn_loss = aux_loss + dyn_loss
        total_dyn_loss.backward()
        self.dynamics_optimizer.step()

        acs, features, next_features = self.update_auxiliary_task(
            acs, obs, last_obs, return_next_features=not self.use_disagreement
        )

        self.policy_optimizer.zero_grad()
        policy_loss, loss_info = self.backprop_loss(acs, features, next_features)
        policy_loss.backward()
        self.policy_optimizer.step()

        loss_info.update(aux_loss_info)
        loss_info.update(dyn_loss_info)
        loss_info["loss/total_loss"] = policy_loss + total_dyn_loss

        return loss_info

    def log_post_update(self, info):

        info["performance/buffer_external_rewards"] = torch.sum(
            self.rollout.buf_ext_rewards
        )
        # This is especially for the robot_arm environment because the touch sensor
        # magnitude can vary a lot.
        info["performance/buffer_external_rewards_mean"] = torch.mean(
            self.rollout.buf_ext_rewards
        )
        info["performance/buffer_external_rewards_present"] = torch.mean(
            (self.rollout.buf_ext_rewards > 0).float()
        )
        info["run/n_updates"] = self.n_updates
        info.update({
            dn: (np.mean(dvs) if len(dvs) > 0 else 0)
            for (dn, dvs) in self.rollout.statlists.items()
        })
        info.update(self.rollout.stats)
        if "states_visited" in info:
            info.pop("states_visited")
        tnow = time.time()
        info["run/updates_per_second"] = 1.0 / (tnow - self.t_last_update)
        self.total_secs = tnow - self.t_start + self.time_trained_so_far
        info["run/total_secs"] = self.total_secs
        info["run/tps"] = self.rollout.nsteps * self.nenvs / (tnow - self.t_last_update)
        self.t_last_update = tnow

        return info

    def auxiliary_loss(self):
        # Get the loss and variance of the feature model
        aux_loss = torch.mean(self.auxiliary_task.get_loss())
        # Take variance over steps -> [feature_dim] vars -> average
        # This is the average variance in a feature over time
        feature_var = torch.mean(torch.var(self.auxiliary_task.features, [0, 1]))
        feature_var_2 = torch.mean(torch.var(self.auxiliary_task.features, [2]))
        return aux_loss, {
            "phi/feature_var_ax01": feature_var,
            "phi/feature_var_ax2": feature_var_2,
            "loss/auxiliary_task": aux_loss
        }

    def update_auxiliary_task(self, acs, obs, last_obs, return_next_features=True):
        # Update the auxiliary task
        self.auxiliary_task.policy.update_features(obs, acs)
        self.auxiliary_task.update_features(obs, last_obs)

        # Gather the data from auxiliary task
        features = self.auxiliary_task.features.detach()
        ac = self.auxiliary_task.ac
        next_features = None
        if return_next_features:
            next_features = self.auxiliary_task.next_features.detach()

        return ac, features, next_features

    def calculate_disagreement(self, acs, features, next_features):
        """ If next features is defined, return prediction error.
        Otherwise returns predictions i.e. dynamics model last layer output
        """

        # Get predictions
        predictions = []
        # Get output from all dynamics models (featurewise)
        for idx, dynamics in enumerate(self.dynamics_list):
            print(f"Running dynamics model: {idx+1}/{len(self.dynamics_list)}")
            # If using disagreement, prediiction is the next state
            # shape=[num_dynamics, num_envs, n_steps_per_seg, feature_dim]
            prediction = dynamics.get_predictions(ac=acs, features=features)
            if next_features is not None:
                # If not using disagreement, prediction is the error
                # shape=[num_dynamics, num_envs, n_steps_per_seg]
                prediction = dynamics.get_prediction_error(prediction, next_features)
            predictions.append(prediction)

        # Get variance over dynamics models
        disagreement = torch.var(torch.stack(predictions), axis=0)
        return disagreement

    def calculate_rewards(self, acs, obs, last_obs):
        """Calculates the reward from the output of the dynamics models and the external
        rewards.
        """

        acs, features, next_features = self.update_auxiliary_task(
            acs, obs, last_obs, return_next_features=not self.use_disagreement
        )

        disagreement = self.calculate_disagreement(acs, features, next_features)
        disagreement_reward = torch.mean(disagreement, axis=-1)
        return disagreement_reward

    def backprop_loss(self, acs, features, next_features):
        """
        TODO: parallelize this loop! Can use Ray, torch.mp, etc
        TODO: This is what takes longer. Could we just store the disagreement in the
        buffer during rollout?
        """

        disagreement = self.calculate_disagreement(acs, features, next_features)
        # Loss is minimized, and we need to maximize variance, so using the inverse
        loss = 1 / torch.mean(disagreement)
        return loss, {"loss/backprop_through_reward_loss": loss}

    def dynamics_loss(self, acs, features, next_features):
        dyn_prediction_loss = 0

        for idx, dynamic in enumerate(self.dynamics_list):
            print(f"Getting dynamics model prediction error: {idx+1}/{len(self.dynamics_list)}")  # noqa: E501
            # Put features into dynamics model and get partial loss (dropout)
            dyn_prediction_loss += torch.mean(dynamic.get_predictions_partial(
                acs, features, next_features
            ))

        # Divide by number of models so the number of dynamic models doesn't impact
        # the total loss. Multiply by a weight that can be defined by the user
        dyn_prediction_loss /= len(self.dynamics_list)
        dyn_prediction_loss *= self.dyn_loss_weight

        return dyn_prediction_loss, {
            "loss/dyn_prediction_loss": dyn_prediction_loss
        }

    def ppo_loss(self, acs, neglogprobs, advantages, returns, *args):

        # Reshape actions and put in tensor
        acs = flatten_dims(acs, len(self.ac_space.shape))
        # Get the negative log probs of the actions under the policy
        neglogprobs_new = self.policy.pd.neglogp(acs.type(torch.LongTensor))
        # Get the entropy of the current policy
        entropy = torch.mean(self.policy.pd.entropy())
        # Get the value estimate of the policies value head
        vpred = self.policy.vpred
        # Calculate the msq difference between value estimate and return
        vf_loss = 0.5 * torch.mean((vpred.squeeze() - returns.detach()) ** 2)
        # Put old neglogprobs from buffer into tensor
        neglogprobs_old = flatten_dims(neglogprobs, 0)
        # Calculate exp difference between old nlp and neglogprobs_new
        # neglogprobs: negative log probability of the action (old)
        # neglogprobs_new: negative log probability of the action (new)
        ratio = torch.exp(neglogprobs_old.detach() - neglogprobs_new.squeeze())
        # Put advantages and negative advantages into tensors
        advantages = flatten_dims(advantages.detach(), 0)
        neg_advantages = -advantages
        # Calculate policy gradient loss. Once multiplied with original ratio
        # between old and new policy probs (1 if identical) and once with
        # clipped ratio.
        policy_gradient_losses1 = neg_advantages * ratio
        policy_gradient_losses2 = neg_advantages * torch.clamp(
            ratio, min=1.0 - self.cliprange, max=1.0 + self.cliprange
        )
        # Get the bigger of the two losses
        policy_gradient_loss_surr = torch.max(
            policy_gradient_losses1, policy_gradient_losses2
        )
        # Get the average policy gradient loss
        policy_gradient_loss = torch.mean(policy_gradient_loss_surr)

        # Get an approximation of the kl-difference between old and new policy
        # probabilities (mean squared difference)
        approx_kl_divergence = 0.5 * torch.mean(
            (neglogprobs_old - neglogprobs_new.squeeze()) ** 2
        )
        # Get the fraction of times that the policy gradient loss was clipped
        clipfrac = torch.mean(
            (torch.abs(policy_gradient_losses2 - policy_gradient_loss_surr) > 1e-6)
            .float()
        )

        # Multiply the policy entropy with the entropy coeficient
        entropy_loss = (-self.entropy_coef) * entropy

        # Calculate the total loss out of the policy gradient loss, the entropy
        # loss (*entropy_coef), the value function loss (*0.5) and feature loss
        # TODO: problem in pg loss and vf loss: Trying to backpropagate a second time
        ppo_loss = policy_gradient_loss + entropy_loss + vf_loss

        return ppo_loss, {
            "ppo/approx_kl_divergence": approx_kl_divergence,
            "ppo/clipfraction": clipfrac,
            "loss/policy_gradient_loss": policy_gradient_loss,
            "loss/value_loss": vf_loss,
            "loss/entropy_loss": entropy_loss,
        }

    def step(self):
        """Collect one rollout and use it to update the networks.

        Returns
        -------
        dict
            Update infos for logging.

        """
        # Collect rollout
        print("--------------------collect rollout------------------------------------")
        acs, obs, last_obs = self.rollout.collect_rollout()

        print("--------------------calculate reward-----------------------------------")
        disagreement_reward = self.calculate_rewards(acs, obs, last_obs)
        self.rollout.update_buffer_pre_step(disagreement_reward)

        print("--------------------gradient steps-------------------------------------")
        update_info = self.learn()

        print("---------------------log statistics-----------------------------------")
        convert_log_to_numpy(update_info)
        self.step_count = self.start_step + self.rollout.step_count

        print("-------------------------done------------------------------------------")

        # Return the update statistics for logging
        return {"update": update_info}

    def get_activation_stats(self, act, name):
        stacked_act = act.view((act.shape[0] * act.shape[1], act.shape[2]))
        num_active = torch.sum(stacked_act > 0, axis=0)

        stats = dict()
        stats[name + "percentage_active_per_frame"] = (
            torch.mean(torch.sum(stacked_act > 0, axis=1) / act.shape[2]) * 100
        )
        stats[name + "percentage_dead"] = (
            torch.sum(num_active == 0) / stacked_act.shape[1] * 100
        )
        # Classically the gini index is defined for positive values (if negative values
        # are included is can be > 1) so we take the abs of activations. For the agent
        # activation this doesn't make a difference since all output of the ReLu is > 0
        # The features are distributed around 0 so here taking the abs has an effect.
        stats[name + "gini_index"] = gini(torch.abs(act))

        return stats, stacked_act


class RewardForwardFilter(object):
    """Discounts reward in the future by gamma and add rewards..

    Parameters
    ----------
    gamma : float
        Discount factor for future rewards.

    Attributes
    ----------
    rewems : array
        rewards so far * discount factor + new rewards.
        shape = [n_envs]

    """

    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def gini(tensor):
    """
    Gini index is a measure of sparsity (https://arxiv.org/abs/0811.4706) where
    an index close to 1 is very sparse and close to 0 has little sparsity.
    """
    tensor = tensor.flatten()
    tensor += 0.0000001  # we want all values to be > 0
    tensor, _ = torch.sort(tensor)
    index = torch.arange(1, tensor.shape[0] + 1)
    n = tensor.shape[0]
    return (torch.sum((2 * index - n - 1) * tensor)) / (n * torch.sum(tensor))
