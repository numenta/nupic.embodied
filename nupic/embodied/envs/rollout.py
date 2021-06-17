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

from collections import defaultdict, deque

import numpy as np
import torch


class Rollout(object):
    """Collect rollouts of experiences in the environments and process them.

    Parameters
    ----------
    ob_space : Space
        Observation space properties (from env.observation_space).
    ac_space : Space
        Action space properties (from env.action_space).
    nenvs : int
        Number of environments used for collecting experiences.
    nsteps_per_seg : int
        Number of steps per rollout segment in each environment. (~like batch size?)
    nsegs_per_env : int
        Number of segments per environment in a rollout..
    nlumps : type
        ..
    envs : [VecEnv]
        List of VecEnvs to use for experience collection.
    policy : object
        CnnPolicy used for action selection.
    internal_reward_coeff : float
        Coefficient for the internal reward (disagreement).
    ext_rew_coeff : float
        Coefficient for the external reward from the environment.
    dynamics_list : [object]
        List of dynamics networks.

    Attributes:
    ----------
    nsteps : int
        nsteps_per_seg * nsegs_per_env.
    lump_stride : int
        TODO.
    reward_function : lambda
        reward function specifying how to combine internal and external rewards.
    buf_vpreds : array
        Buffer of value estimates.
    buf_neglogprobs : array
        Buffer of negative log probabilities.
    buf_rewards : array
        Buffer of rewards.
    buf_ext_rewards : array
        Buffer of external rewards.
    buf_acs : array
        Buffer of actions.
    buf_obs : array
        Buffer of observations.
    buf_obs_last : array
        Buffer of last observations.
    buf_dones : array
        Buffer of 'dones'.
    buf_done_last : array
        Buffer of last 'dones'.
    buf_vpred_last : array
        Buffer of last value estimates.
    env_results : List
        Outputs from the environment (observations, rewards, done, info).
    internal_reward : array
        Internal rewards.
    statlists : dict
        Dictionary with run statistics.

    """

    def __init__(
        self,
        ob_space,
        ac_space,
        nenvs,
        nsteps_per_seg,
        nsegs_per_env,
        nlumps,
        envs,
        policy,
        int_rew_coeff,
        ext_rew_coeff,
        dynamics_list,
    ):
        self.nenvs = nenvs
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.nlumps = nlumps
        self.lump_stride = nenvs // self.nlumps
        self.envs = envs
        self.policy = policy
        self.dynamics_list = dynamics_list

        # Define the reward function as a weighted combination of internal and (clipped)
        # external rewards.
        self.reward_function = (
            lambda ext_rew, internal_reward: ext_rew_coeff * np.clip(ext_rew, -1.0, 1.0)
            + int_rew_coeff * internal_reward
        )

        # Initialize buffer
        self.buf_vpreds = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_neglogprobs = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_rewards = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_ext_rewards = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_acs = np.empty(
            (nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype
        )
        self.buf_obs = np.empty(
            (nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype
        )
        self.buf_obs_last = np.empty(
            (nenvs, self.nsegs_per_env, *self.ob_space.shape), np.float32
        )
        self.buf_dones = np.zeros((nenvs, self.nsteps), np.float32)
        self.buf_done_last = self.buf_dones[:, 0, ...].copy()
        self.buf_vpred_last = self.buf_vpreds[:, 0, ...].copy()

        self.buf_acts_features = np.empty(
            (nenvs, self.nsteps_per_seg, self.policy.feature_dim)
        )
        self.buf_acts_pi = np.empty(
            (nenvs, self.nsteps_per_seg, self.policy.feature_dim)
        )

        self.env_results = [None] * self.nlumps
        self.internal_reward = np.zeros((nenvs,), np.float32)

        self.statlists = defaultdict(lambda: deque([], maxlen=100))
        self.stats = defaultdict(float)
        self.best_ext_return = None

        self.step_count = 0

    def collect_rollout(self):
        """Steps through environment, calculates reward and update info."""
        self.ep_infos_new = []
        for _ in range(self.nsteps):
            self.rollout_step()
        self.calculate_reward()
        self.update_info()

    def load_from_buffer(self, idxs):
        """
        Loads relevant buffer information to be used by the agent update step
        Note: negative log probabilities are the action probabilities from pi
        """
        acs = self.buf_acs[idxs]
        rews = self.buf_rewards[idxs]
        neglogprobs = self.buf_neglogprobs[idxs]
        obs = self.buf_obs[idxs]
        last_obs = self.buf_obs_last[idxs]
        return acs, rews, neglogprobs, obs, last_obs

    def calculate_backprop_loss(self):
        """
        Calculates the reward from the output of teh dynamics models and the external
        rewards.

        """

        pred_features = []
        # Get output from all dynamics models (featurewise)
        # shape=[num_dynamics, num_envs, n_steps_per_seg, feature_dim]

        # Forward pass per dynamic model
        for dynamics in self.dynamics_list:
            pred_features.append(
                dynamics.predict_features(
                    obs=self.buf_obs, last_obs=self.buf_obs_last, acs=self.buf_acs
                )
            )

        # Get variance over dynamics models
        # shape=[n_envs, n_steps_per_seg, feature_dim]  # 64, 128, 512
        # n_dynamic_models, n_evns, n_steps_per_seg, feature_dim
        disagreement = torch.var(torch.stack(pred_features), axis=0)
        # n_envs, n_steps_per_seg, feature_dim
        loss = -torch.mean(disagreement)
        return loss

    def calculate_reward(self):
        """Calculates the reward from the output of teh dynamics models and the external
        rewards.

        """
        net_output = []
        if self.dynamics_list[0].use_disagreement:
            # Get output from all dynamics models (featurewise)
            # shape=[num_dynamics, num_envs, n_steps_per_seg, feature_dim]

            for dynamics in self.dynamics_list:
                net_output.append(
                    dynamics.calculate_loss(
                        obs=self.buf_obs, last_obs=self.buf_obs_last, acs=self.buf_acs
                    )
                )
            # Get variance over dynamics models
            # shape=[n_envs, n_steps_per_seg, feature_dim]
            disagreement = np.var(net_output, axis=0)
            # Get reward by mean along features
            # shape=[n_envs, n_steps_per_seg]
            disagreement_reward = np.mean(disagreement, axis=-1)
        else:
            # Get loss from all dynamics models (difference between dynamics output and
            # the features of the next state)
            # shape=[num_dynamics, num_envs, n_steps_per_seg]
            for dynamics in self.dynamics_list:
                net_output.append(
                    dynamics.calculate_loss(
                        obs=self.buf_obs, last_obs=self.buf_obs_last, acs=self.buf_acs
                    )
                )
            # Get the variance of the dynamics loss over dynamic models
            # shape=[n_envs, n_steps_per_seg]
            disagreement_reward = np.var(net_output, axis=0)
        # Fill reward buffer with the new rewards
        self.buf_rewards[:] = self.reward_function(
            internal_reward=disagreement_reward, ext_rew=self.buf_ext_rewards
        )

    def rollout_step(self):
        """Take a step in the environment and fill the buffer with all infos."""
        t = self.step_count % self.nsteps
        s = t % self.nsteps_per_seg
        for lump in range(self.nlumps):  # TODO: What is lump? default=1
            # Get results from environment step (if first step, reset env)
            obs, prevrews, news, infos = self.env_get(lump)
            # Extract episode infos
            for info in infos:
                epinfo = info.get("episode", {})
                mzepinfo = info.get("mz_episode", {})
                retroepinfo = info.get("retro_episode", {})
                epinfo.update(mzepinfo)
                epinfo.update(retroepinfo)
                if epinfo:
                    if "n_states_visited" in info:
                        epinfo["n_states_visited"] = info["n_states_visited"]
                        epinfo["states_visited"] = info["states_visited"]
                    self.ep_infos_new.append((self.step_count, epinfo))

            # Get actions, value estimates and nedlogprobs for obs from policy
            acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)

            # Execute the policies actions in the environments
            self.env_step(lump, acs)
            # Fill the buffer
            sli = slice(lump * self.lump_stride, (lump + 1) * self.lump_stride)
            self.buf_obs[sli, t] = obs
            self.buf_dones[sli, t] = news
            self.buf_vpreds[sli, t] = vpreds
            self.buf_neglogprobs[sli, t] = nlps
            self.buf_acs[sli, t] = acs
            if t > 0:
                self.buf_ext_rewards[sli, t - 1] = prevrews

            # Fill buffer with hidden activations (used for stats logging)
            self.buf_acts_features[
                sli, t
            ] = self.policy.flat_features.cpu().data.numpy()
            self.buf_acts_pi[sli, t] = self.policy.hidden_pi.cpu().data.numpy()

        self.step_count += 1
        if s == self.nsteps_per_seg - 1:
            # Get the experiences for the last step of the segment.
            for lump in range(self.nlumps):
                sli = slice(lump * self.lump_stride, (lump + 1) * self.lump_stride)
                nextobs, ext_rews, nextnews, _ = self.env_get(lump)
                self.buf_obs_last[sli, t // self.nsteps_per_seg] = nextobs
                if t == self.nsteps - 1:
                    self.buf_done_last[sli] = nextnews
                    self.buf_ext_rewards[sli, t] = ext_rews
                    _, self.buf_vpred_last[sli], _ = self.policy.get_ac_value_nlp(
                        nextobs
                    )

    def update_info(self):
        """If there is episode info (like stats at the end of an episode) save them."""
        all_ep_infos = self.ep_infos_new
        if all_ep_infos:
            all_ep_infos = [i_[1] for i_ in all_ep_infos]  # remove the step_count
            keys_ = all_ep_infos[0].keys()
            all_ep_infos = {k: [i[k] for i in all_ep_infos] for k in keys_}

            self.statlists["performance/episode_reward"].extend(all_ep_infos["r"])
            self.stats["performance/eprew_recent"] = np.mean(all_ep_infos["r"])
            self.statlists["performance/episode_length"].extend(all_ep_infos["l"])
            self.stats["run/episode_count"] += len(all_ep_infos["l"])
            self.stats["run/num_timesteps"] += sum(all_ep_infos["l"])

            current_max = np.max(all_ep_infos["r"])
        else:
            current_max = None
        self.ep_infos_new = []

        if current_max is not None:
            if (self.best_ext_return is None) or (current_max > self.best_ext_return):
                self.best_ext_return = current_max
        self.current_max = current_max

    def env_step(self, lump, acs):
        """Take asynchronous steps in the environments.

        Parameters
        ----------
        lump : type
            todo.
        acs : array
            Actions that should be executed.

        """
        self.envs[lump].step_async(acs)
        self.env_results[lump] = None

    def env_get(self, lump):
        """Return the observations after taking a step in the environment.

        Parameters
        ----------
        lump : type
            Description of parameter `lump`..

        Returns
        -------
        List
            List of observations, prevrews, news, infos

        """
        if self.step_count == 0:
            # Reset the environment if the step count is zero
            ob = self.envs[lump].reset()
            print("env reset")
            out = self.env_results[lump] = (
                ob,
                None,
                np.ones(self.lump_stride, bool),
                {},
            )
        else:
            if self.env_results[lump] is None:
                # In there are no results yet, wait.
                out = self.env_results[lump] = self.envs[lump].step_wait()
            else:
                out = self.env_results[lump]
        return out
