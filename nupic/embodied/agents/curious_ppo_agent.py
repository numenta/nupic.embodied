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
    exp_name : str
        Name of the experiment (used for video logging).. currently not used
    vlog_freq : int
        After how many steps should a video of the training be logged.
    dynamics_list : [Dynamics]
        List of dynamics models to use for internal reward calculation.

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
        exp_name,  # TODO: not being used, delete it?
        vlog_freq,
        debugging,
        dynamics_list,
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
        self.backprop_through_reward = backprop_through_reward

    def start_interaction(self, env_fns, dynamics_list, nlump=1):
        """Set up environments and initialize everything.

        Parameters
        ----------
        env_fns : [envs]
            List of environments (functions), optionally with wrappers etc.
        dynamics_list : [Dynamics]
            List of dynamics models.
        nlump : int
            ..

        """
        # Specify parameters that should be optimized
        # auxiliary task params is the same for all dynamic models
        param_list = [
            *self.policy.param_list,
            *self.dynamics_list[0].auxiliary_task.param_list,
        ]
        for dynamic in self.dynamics_list:
            param_list.extend(dynamic.param_list)

        # Initialize the optimizer
        if self.backprop_through_reward:
            self.optimizer = torch.optim.Adam(self.policy.param_list, lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(param_list, lr=self.lr)

        # Set gradients to zero
        self.optimizer.zero_grad()

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
            dynamics_list=dynamics_list,
        )

        # Initialize replay buffers for advantages and returns of each rollout
        # TODO: standardize to torch
        self.buf_advantages = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_returns = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        if self.normrew:  # if normalize reward, defaults to True
            # Sum up and discount rewards
            self.reward_forward_filter = RewardForwardFilter(self.gamma)
            # Initialize running mean and std tracker
            self.reward_stats = RunningMeanStd()

        self.step_count = 0
        self.start_step = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self):
        """Close environments when stopping."""
        for env in self.envs:
            env.close()

    def calculate_advantages(self, rews, use_done, gamma, lam, normalize):
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

            # The value etimate of the next time step
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

    def init_info_dict(self):
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
            self.rollout.buf_vpreds.ravel(), self.buf_returns.ravel()
        )
        info["ppo/reward_mean"] = np.mean(self.rollout.buf_rewards)

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
                stacked_act_feat
            )
            info["activations_hidden/raw_act_distribution"] = wandb.Histogram(
                stacked_act_pi
            )

            info["ppo/action_distribution"] = wandb.Histogram(
                self.rollout.buf_acs.flatten()
            )

            if self.vLogFreq >= 0 and self.n_updates % self.vLogFreq == 0:
                print(str(self.n_updates) + " updates - logging video.")
                # Reshape images such that they have shape [time,channels,width,height]
                sample_video = np.moveaxis(self.rollout.buf_obs[0], 3, 1)
                # Log buffer video from first env
                info["observations"] = wandb.Video(sample_video, fps=12, format="gif")

        return info

    def collect_rewards(self, normalize=False):
        if normalize:
            discounted_rewards = np.array(
                [
                    self.reward_forward_filter.update(rew)
                    for rew in self.rollout.buf_rewards.T
                ]
            )
            # rewards_mean, rewards_std, rewards_count
            rewards_mean, rewards_std, rewards_count = mpi_moments(
                discounted_rewards.ravel()
            )
            # reward forward filter running mean std
            self.reward_stats.update_from_moments(
                rewards_mean, rewards_std ** 2, rewards_count
            )
            return self.rollout.buf_rewards / np.sqrt(self.reward_stats.var)
        else:
            # Copy directly from buff rewards
            return np.copy(self.rollout.buf_rewards)


    def update(self):
        """Calculate losses and update parameters based on current rollout.

        Returns
        -------
        info
            Dictionary of infos about the current update and training statistics.

        """
        rews = self.collect_rewards(normalize=self.normrew)

        # Calculate advantages using the current rewards and value estimates
        self.calculate_advantages(
            rews=rews, use_done=self.use_done, gamma=self.gamma, lam=self.lam
        )

        # TODO: we are logging buf_advantages before normalizing, is that correct?
        info = self.init_info_dict()

        to_report = Counter()

        if self.normadv:  # defaults to True
            # normalize advantages
            m, s = get_mean_and_std(self.buf_advantages)
            self.buf_advantages = (self.buf_advantages - m) / (s + 1e-7)
        # Set update hyperparameters
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        # Update the networks & get losses for nepochs * nminibatches
        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                minibatch_envinds = envinds[start:end]  # minibatch environment indexes
                # Get rollout experiences for current minibatch
                acs = self.rollout.buf_acs[minibatch_envinds]
                rews = self.rollout.buf_rewards[minibatch_envinds]
                neglogprobs = self.rollout.buf_neglogprobs[
                    minibatch_envinds
                ]  # negative log probabilities (action probabilities from pi)
                obs = self.rollout.buf_obs[minibatch_envinds]
                returns = self.buf_returns[minibatch_envinds]
                advantages = self.buf_advantages[minibatch_envinds]
                last_obs = self.rollout.buf_obs_last[minibatch_envinds]

                # Update features of the policy network to minibatch obs and acs
                self.policy.update_features(obs, acs)

                # Update features of the auxiliary network to minibatch obs and acs
                # Using first element in dynamics list is sufficient bc all dynamics
                # models have the same auxiliary task model and features
                # TODO: should the feature model be independent of dynamics?
                self.dynamics_list[0].auxiliary_task.update_features(obs, last_obs)
                # Get the loss and variance of the feature model
                aux_loss = torch.mean(self.dynamics_list[0].auxiliary_task.get_loss())
                # Take variance over steps -> [feature_dim] vars -> average
                # This is the average variance in a feature over time
                feature_var = torch.mean(
                    torch.var(self.dynamics_list[0].auxiliary_task.features, [0, 1])
                )
                feature_var_2 = torch.mean(
                    torch.var(self.dynamics_list[0].auxiliary_task.features, [2])
                )

                # disagreement = []
                dyn_prediction_loss = []
                # Loop through dynamics models
                for dynamic in self.dynamics_list:
                    # Get the features of the observations in the dynamics model (just
                    # gets features from the auxiliary model)
                    dynamic.update_features()
                    # Put features into dynamics model and get loss
                    # (if use_disagreement just returns features, therfor here the
                    # partial loss is used for optimizing and loging)
                    # disagreement.append(torch.mean(np.var(dynamic.get_loss(),axis=0)))

                    # Put features into dynamics model and get partial loss (dropout)
                    dyn_prediction_loss.append(torch.mean(dynamic.get_loss_partial()))

                # Reshape actions and put in tensor
                acs = torch.tensor(flatten_dims(acs, len(self.ac_space.shape))).to(
                    self.device
                )
                # Get the negative log probs of the actions under the policy
                neglogprobs_new = self.policy.pd.neglogp(acs)
                # Get the entropy of the current policy
                entropy = torch.mean(self.policy.pd.entropy())
                # Get the value estimate of the policies value head
                vpred = self.policy.vpred
                # Calculate the msq difference between value estimate and return
                vf_loss = 0.5 * torch.mean(
                    (vpred.squeeze() - torch.tensor(returns).to(self.device)) ** 2
                )
                # Put old neglogprobs from buffer into tensor
                neglogprobs_old = torch.tensor(flatten_dims(neglogprobs, 0)).to(
                    self.device
                )
                # Calculate exp difference between old nlp and neglogprobs_new
                # neglogprobs: negative log probability of the action (old)
                # neglogprobs_new: negative log probability of the action (new)
                ratio = torch.exp(neglogprobs_old - neglogprobs_new.squeeze())
                # Put advantages and negative advantages into tensors
                advantages = flatten_dims(advantages, 0)
                neg_advantages = torch.tensor(-advantages).to(self.device)
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
                    (
                        torch.abs(policy_gradient_losses2 - policy_gradient_loss_surr)
                        > 1e-6
                    ).float()
                )

                # Multiply the policy entropy with the entropy coeficient
                entropy_loss = (-self.entropy_coef) * entropy

                # Calculate the total loss out of the policy gradient loss, the entropy
                # loss (*entropy_coef), the value function loss (*0.5) and feature loss
                total_loss = policy_gradient_loss + entropy_loss + vf_loss + aux_loss
                for i in range(len(dyn_prediction_loss)):
                    # add the loss of each of the dynamics networks to the total loss
                    total_loss = total_loss + dyn_prediction_loss[i]
                # propagate the loss back through the networks
                total_loss.backward()
                self.optimizer.step()
                # set the gradients back to zero
                self.optimizer.zero_grad()

                # Log statistics (divide by nminibatchs * nepochs because we add the
                # loss in these two loops.)
                to_report["loss/total_loss"] += total_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report[
                    "loss/policy_gradient_loss"
                ] += policy_gradient_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["loss/value_loss"] += vf_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["loss/entropy_loss"] += entropy_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report[
                    "ppo/approx_kl_divergence"
                ] += approx_kl_divergence.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["ppo/clipfraction"] += clipfrac.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["phi/feature_var_ax01"] += feature_var.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["phi/feature_var_ax2"] += feature_var_2.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["loss/auxiliary_task"] += aux_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["loss/dynamic_loss"] += np.sum(
                    [e.cpu().data.numpy() for e in dyn_prediction_loss]
                ) / (self.nminibatches * self.nepochs)

        info.update(to_report)
        self.n_updates += 1
        info["performance/buffer_external_rewards"] = np.sum(
            self.rollout.buf_ext_rewards
        )
        # This is especially for the robot_arm environment because the touch sensor
        # magnitude can vary a lot.
        info["performance/buffer_external_rewards_mean"] = np.mean(
            self.rollout.buf_ext_rewards
        )
        info["performance/buffer_external_rewards_present"] = np.mean(
            self.rollout.buf_ext_rewards > 0
        )
        info["run/n_updates"] = self.n_updates
        info.update(
            {
                dn: (np.mean(dvs) if len(dvs) > 0 else 0)
                for (dn, dvs) in self.rollout.statlists.items()
            }
        )
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

    def step(self):
        """Collect one rollout and use it to update the networks.

        Returns
        -------
        dict
            Update infos for logging.

        """
        # Collect rollout
        self.rollout.collect_rollout()

        # Calculate reward or loss
        if self.backprop_through_reward:
            loss = self.backprop_gradient_step()
            # TODO: break the update function into two, one only to log
            update_info = dict(backprop_loss=loss)
        else:
            # Calculate losses and backpropagate them through the networks
            update_info = self.update()

        # Update stepcount
        self.step_count = self.start_step + self.rollout.step_count
        # Return the update statistics for logging
        return {"update": update_info}

    def backprop_gradient_step(self):
        self.optimizer.zero_grad()
        loss = self.rollout.calculate_backprop_loss()
        print(f"Loss from the backprop: {loss:.4f}")
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_activation_stats(self, act, name):
        stacked_act = np.reshape(act, (act.shape[0] * act.shape[1], act.shape[2]))
        num_active = np.sum(stacked_act > 0, axis=0)

        def gini(array):
            """
            Gini index is a measure of sparsity (https://arxiv.org/abs/0811.4706) where
            an index close to 1 is very sparse and close to 0 has little sparsity.
            """
            array = array.flatten()
            array += 0.0000001  # we want all values to be > 0
            array = np.sort(array)
            index = np.arange(1, array.shape[0] + 1)
            n = array.shape[0]
            return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

        stats = dict()
        stats[name + "percentage_active_per_frame"] = (
            np.mean(np.sum(stacked_act > 0, axis=1) / act.shape[2]) * 100
        )
        stats[name + "percentage_dead"] = (
            np.sum(num_active == 0) / stacked_act.shape[1] * 100
        )
        # Classically the gini index is defined for positive values (if negative values
        # are included is can be > 1) so we take the abs of activations. For the agent
        # activation this doesn't make a difference since all output of the ReLu is > 0
        # The features are distributed around 0 so here taking the abs has an effect.
        stats[name + "gini_index"] = gini(np.abs(act))

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
