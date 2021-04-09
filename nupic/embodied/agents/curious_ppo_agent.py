import time
from nupic.embodied.utils.model_parts import flatten_dims
import torch
import numpy as np
from nupic.embodied.utils.mpi import mpi_moments

from nupic.embodied.envs.rollout import Rollout
from nupic.embodied.utils.utils import (
    get_mean_and_std,
    explained_variance,
    RunningMeanStd,
)
from nupic.embodied.envs.vec_env import ShmemVecEnv as VecEnv

from collections import Counter


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
    stochpol : CnnPolicy
        Stochastic policy to use for action selection.
    ent_coef : float
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
    use_news : bool
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
    expName : str
        Name of the experiment (used for video logging).. currently not used
    vLogFreq : int
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
        *,
        scope,
        device,
        ob_space,
        ac_space,
        stochpol,
        ent_coef,
        gamma,
        lam,
        nepochs,
        lr,
        cliprange,
        nminibatches,
        normrew,
        normadv,
        use_news,
        ext_coeff,
        int_coeff,
        nsteps_per_seg,
        nsegs_per_env,
        expName,
        vLogFreq,
        dynamics_list,
    ):
        self.dynamics_list = dynamics_list
        self.n_updates = 0
        self.scope = scope
        self.device = device
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.stochpol = stochpol
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
        self.use_news = use_news
        self.ent_coef = ent_coef
        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff
        self.vLogFreq = vLogFreq
        # TODO: Add saving with expName & vLog Freq video saving

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
            *self.stochpol.param_list,
            *self.dynamics_list[0].auxiliary_task.param_list
        ]
        for dynamic in self.dynamics_list:
            param_list.extend(dynamic.param_list)

        # Initialize the optimizer
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
            policy=self.stochpol,
            int_rew_coeff=self.int_coeff,
            ext_rew_coeff=self.ext_coeff,
            dynamics_list=dynamics_list,
        )

        # Initialize replay buffers for advantages and returns of each rollout
        # TODO: standardize to torch
        self.buf_advs = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        if self.normrew:  # if normalize reward, defaults to True
            # Sum up and discount rewards
            self.rff = RewardForwardFilter(self.gamma)
            # Initialize running mean and std tracker
            self.rff_rms = RunningMeanStd()

        self.step_count = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self):
        """Close environments when stopping."""
        for env in self.envs:
            env.close()

    def calculate_advantages(self, rews, use_news, gamma, lam):
        """Calculate advantages from the rewards.

        Parameters
        ----------
        rews : array
            rewards. shape = [n_envs, n_steps]
        use_news : bool
            Whether to use news (which are the done infos from the environment).
        gamma : float
            Discount factor for the rewards.
        lam : float
            Generalized advantage estimator smoothing parameter.

        """
        nsteps = self.rollout.nsteps
        lastgaelam = 0
        for t in range(nsteps - 1, -1, -1):  # going backwards from last step in seg
            # Is the next step a new run (done=True)?
            nextnew = (
                self.rollout.buf_news[:, t + 1]
                if t + 1 < nsteps
                else self.rollout.buf_new_last
            )
            if not use_news:
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
            self.buf_advs[:, t] = lastgaelam = (
                delta + gamma * lam * nextnotnew * lastgaelam
            )
        # Update return buffer (advantages + value estimates)
        self.buf_rets[:] = self.buf_advs + self.rollout.buf_vpreds

    def update(self):
        """Calculate losses and update parameters based on current rollout.

        Returns
        -------
        info
            Dictionary of infos about the current update and training statistics.

        """
        if self.normrew:
            # Normalize the rewards using the running mean and std
            rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])
            # rewards_mean, rewards_std, rewards_count
            rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
            # reward forward filter running mean std
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = self.rollout.buf_rews / np.sqrt(self.rff_rms.var)
        else:
            rews = np.copy(self.rollout.buf_rews)

        # Calculate advantages using the current rewards and value estimates
        self.calculate_advantages(
            rews=rews, use_news=self.use_news, gamma=self.gamma, lam=self.lam
        )

        # Initialize and update the info dict for logging
        info = dict()
        info["ppo/advantage_mean"] = self.buf_advs.mean()
        info["ppo/advantage_std"] = self.buf_advs.std()
        info["ppo/return_mean"] = self.buf_rets.mean()
        info["ppo/return_std"] = self.buf_rets.std()
        info["ppo/value_est_mean"] = self.rollout.buf_vpreds.mean()
        info["ppo/value_est_std"] = self.rollout.buf_vpreds.std()
        info["ppo/explained_variance"] = explained_variance(
            self.rollout.buf_vpreds.ravel(), self.buf_rets.ravel()
        )
        info["ppo/reward_mean"] = np.mean(self.rollout.buf_rews)

        if self.rollout.best_ext_ret is not None:
            info["performance/best_ext_return"] = self.rollout.best_ext_ret

        to_report = Counter()

        if self.normadv:  # defaults to True
            # normalize advantages
            m, s = get_mean_and_std(self.buf_advs)
            self.buf_advs = (self.buf_advs - m) / (s + 1e-7)
        # Set update hyperparameters
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        # Update the networks & get losses for nepochs * nminibatches
        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]  # minibatch environment indexes
                # Get rollout experiences for current minibatch
                acs = self.rollout.buf_acs[mbenvinds]
                rews = self.rollout.buf_rews[mbenvinds]
                nlps = self.rollout.buf_nlps[mbenvinds]  # negative log probabilities (action probabilities from pi)
                obs = self.rollout.buf_obs[mbenvinds]
                rets = self.buf_rets[mbenvinds]
                advs = self.buf_advs[mbenvinds]
                last_obs = self.rollout.buf_obs_last[mbenvinds]

                # Update features of the policy network to minibatch obs and acs
                self.stochpol.update_features(obs, acs)

                # Update features of the auxiliary network to minibatch obs and acs
                # Using first element in dynamics list is sufficient bc all dynamics
                # models have the same auxiliary task model and features
                # TODO: should the feature model be independent of dynamics?
                self.dynamics_list[0].auxiliary_task.update_features(obs, last_obs)
                # Get the loss and variance of the feature model
                feat_loss = torch.mean(self.dynamics_list[0].auxiliary_task.get_loss())
                # Take variance over steps -> [feat_dim] vars -> average
                # This is the average variance in a feature over time
                feat_var = torch.mean(
                    torch.var(self.dynamics_list[0].auxiliary_task.features, [0, 1])
                )
                feat_var_2 = torch.mean(
                    torch.var(self.dynamics_list[0].auxiliary_task.features, [2])
                )

                # dyn_loss = []
                dyn_partial_loss = []
                # Loop through dynamics models
                for dynamic in self.dynamics_list:
                    # Get the features of the observations in the dynamics model (just
                    # gets features from the auxiliary model)
                    dynamic.update_features()
                    # Put features into dynamics model and get loss
                    # (if var_output just returns fetaures, therfor here the partial
                    # loss is used for optimizing and loging)
                    # dyn_loss.append(torch.mean(dynamic.get_loss()))

                    # Put features into dynamics model and get partial loss (dropout)
                    dyn_partial_loss.append(torch.mean(dynamic.get_loss_partial()))

                # Reshape actions and put in tensor
                acs = torch.tensor(flatten_dims(acs, len(self.ac_space.shape))).to(
                    self.device
                )
                # Get the negative log probs of the actions under the policy
                neglogpac = self.stochpol.pd.neglogp(acs)
                # Get the entropy of the current policy
                entropy = torch.mean(self.stochpol.pd.entropy())
                # Get the value estimate of the policies value head
                vpred = self.stochpol.vpred
                # Calculate the msq difference between value estimate and return
                vf_loss = 0.5 * torch.mean(
                    (vpred.squeeze() - torch.tensor(rets).to(self.device)) ** 2
                )
                # Put old nlps from buffer into tensor
                nlps = torch.tensor(flatten_dims(nlps, 0)).to(self.device)
                # Calculate exp difference between old nlp and neglogpac
                # nlps: negative log probability of the action (old)
                # neglogpac: negative log probability of the action (new)
                ratio = torch.exp(nlps - neglogpac.squeeze())
                # Put advantages and negative advs into tensors
                advs = flatten_dims(advs, 0)
                negadv = torch.tensor(-advs).to(self.device)
                # Calculate policy gradient loss. Once multiplied with original ratio
                # between old and new policy probs (1 if identical) and once with
                # clipped ratio.
                pg_losses1 = negadv * ratio
                pg_losses2 = negadv * torch.clamp(
                    ratio, min=1.0 - self.cliprange, max=1.0 + self.cliprange
                )
                # Get the bigger of the two losses
                pg_loss_surr = torch.max(pg_losses1, pg_losses2)
                # Get the average policy gradient loss
                pg_loss = torch.mean(pg_loss_surr)

                # Get an approximation of the kl-difference between old and new policy
                # probabilities (mean squared difference)
                approxkl = 0.5 * torch.mean((neglogpac.squeeze() - nlps) ** 2)
                # Get the fraction of times that the policy gradient loss was clipped
                clipfrac = torch.mean(
                    (torch.abs(pg_losses2 - pg_loss_surr) > 1e-6).float()
                )

                # Multiply the policy entropy with the entropy coeficient
                ent_loss = (-self.ent_coef) * entropy

                # Calculate the total loss out of the policy gradient loss, the entropy
                # loss (*ent_coef), the value function loss (*0.5) and the feature loss
                total_loss = pg_loss + ent_loss + vf_loss + feat_loss
                for i in range(len(dyn_partial_loss)):
                    # add the loss of each of the dynamics networks to the total loss
                    total_loss = total_loss + dyn_partial_loss[i]
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
                to_report["loss/policy_gradient_loss"] += pg_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["loss/value_loss"] += vf_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["loss/entropy_loss"] += ent_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["ppo/approx_kl_divergence"] += approxkl.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["ppo/clipfraction"] += clipfrac.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["phi/feat_var_ax01"] += feat_var.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["phi/feat_var_ax2"] += feat_var_2.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["loss/auxiliary_task"] += feat_loss.cpu().data.numpy() / (
                    self.nminibatches * self.nepochs
                )
                to_report["loss/dynamic_loss"] += np.sum(
                    [e.cpu().data.numpy() for e in dyn_partial_loss]
                ) / (self.nminibatches * self.nepochs)

        info.update(to_report)
        self.n_updates += 1
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
        info["run/total_secs"] = tnow - self.t_start
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
        # Calculate losses and backpropagate them through the networks
        update_info = self.update()
        # Update stepcount
        self.step_count = self.rollout.step_count
        # Return the update statistics for logging
        return {"update": update_info}


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
