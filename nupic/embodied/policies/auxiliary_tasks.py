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

import numpy as np
import torch

from nupic.embodied.utils.model_parts import (
    flatten_dims,
    small_convnet,
    small_deconvnet,
    unflatten_first_dim,
)


class FeatureExtractor(object):
    """Extract features from high dimensional observations. The default version of this
    has no loss such that the weights are not updated. Therefor we have random features
    which are neverthless usefull because the reduce the observation dimensionality and
    are stable.

    Parameters
    ----------
    policy : CnnPolicy
        The policy of the ppo agent.
    features_shared_with_policy : bool
        Whether to use the policy features for the disagreement module.
    feature_dim : int
        Number of neurons in the last layer of the feature network (or middle of VAE).
    layernormalize : bool
        Whether to normalize output of the last layer.
    scope : str
        torch model scope.
    device: torch.device
        Which device to optimize the model on.

    Attributes
    ----------
    hidden_dim : int
        Number of neurons in the hidden layer.
    ob_space : Space
        Observation space properties (from env.observation_space).
    ac_space : Space
        Action space properties (from env.action_space).
    ob_mean : array
        Mean value of observations collected by a random agent. (Of same size as the
        observations)
    ob_std : float
        Standard deviation of observations collected by a random agent.
    param_list : list
        List of parameters to be optimized.
    features_model : model
        Feature model to use for feature extraction.
    features : array
        Features corresponding to the current observation.
    next_features : array
        Features corresponding to the next observation.
    ac : array
        Current action.
    ob : array
        Current observation.

    """

    def __init__(
        self,
        policy,
        features_shared_with_policy,
        device,
        feature_dim=None,
        layernormalize=None,
        scope="feature_extractor",
    ):
        self.scope = scope
        self.features_shared_with_policy = features_shared_with_policy
        self.feature_dim = feature_dim
        self.layernormalize = layernormalize
        self.policy = policy
        self.hidden_dim = policy.hidden_dim
        self.ob_space = policy.ob_space
        self.ac_space = policy.ac_space
        self.ob_mean = self.policy.ob_mean
        self.ob_std = self.policy.ob_std
        self.device = device

        self.features_shared_with_policy = features_shared_with_policy
        self.param_list = []
        if features_shared_with_policy:
            # No need to nitialize a feature model because it is already initialized in
            # curious_cnn_policy.py
            self.features_model = None
        else:
            # Initialize small convolutional network for feature extraction from images
            self.features_model = small_convnet(
                self.ob_space,
                nonlinear=torch.nn.LeakyReLU,
                feature_dim=self.feature_dim,
                last_nonlinear=None,
                layernormalize=self.layernormalize,
                device=self.device,
            ).to(self.device)
            # Add feature model to optimization parameters
            self.param_list.extend(self.features_model.parameters())
        self.scope = scope

        self.features = None
        self.next_features = None
        self.ac = None
        self.ob = None

    def update_features(self, acs, obs, last_obs):
        """Get features from feature model and set self.features and self.next_features.
        Also sets self.ac to the last actions at end of rollout..

        Parameters
        ----------
        obs : array
            Current observations.
        last_obs : array
            Previous observations.

        """
        self.ob, self.ac = obs, acs

        if self.features_shared_with_policy:
            self.policy.update_features(obs, acs)
            # Get features corresponding with the observation from the policy network
            self.features = self.policy.flat_features
            last_features = self.policy.get_features(last_obs)
        else:
            # Get features corresponding with the observation from the feature network
            self.features = self.get_features(
                obs
            )  # shape=[1, nsteps_per_seg, feature_dim]
            last_features = self.get_features(last_obs)  # shape=[1, 1, feature_dim]
        # concatenate the last and current features
        self.next_features = torch.cat([self.features[:, 1:], last_features], 1)


    def get_features(self, obs):
        """Get features from the feature network.

        Parameters
        ----------
        obs : array
            Observation for which to get features.

        Returns
        -------
        array
            Features of the observations.

        """
        # TODO: refactor - too many shape transformations in obs and act, confusing
        has_timesteps = len(obs.shape) == 5
        if has_timesteps:
            sh = obs.shape  # shape=[1, nsteps, H, W, C]
            obs = flatten_dims(obs, len(self.ob_space.shape))  # shape=[nsteps, H, W, C]
        # Normalize observations
        obs = (obs - self.ob_mean) / self.ob_std
        # Reshape observations, shape=[nsteps, C, H, W]
        obs = obs.permute([i for i in range(len(obs.shape) - 3)] + [-1, -3, -2])
        # Get features from the features_model
        act = self.features_model(obs)

        if has_timesteps:
            act = unflatten_first_dim(act, sh)

        return act

    def get_loss(self, *arg, **kwarg):
        # is 0 because we use no auxiliary task for feature learning
        return torch.tensor(0.0)


class InverseDynamics(FeatureExtractor):
    """Learns feature mapping of the input with the auxiliary task to predict the action
    that was performed between two consecutive states. In the cuirosity paper thsi is
    called the backward loss.

    Parameters
    ----------
    policy : CnnPolicy
        The policy of the ppo agent.
    features_shared_with_policy : bool
        Whether to use the policy features for the disagreement module.
    feature_dim : int
        Number of neurons in the last layer of the feature network (or middle of VAE).
    layernormalize : bool
        Whether to normalize output of the last layer.
    device: torch.device
        Which device to optimize the model on.

    Attributes
    ----------
    fc : torch.nn.Sequential
        Fully-connected network which maps the extracted features to the dimensionality
        of the action space (2 layer with ReLU on the first).
    ac_space : Space
        Action space properties (from env.action_space).
    param_list : list
        List of parameters to be optimized.

    """

    def __init__(
        self,
        policy,
        features_shared_with_policy,
        device,
        feature_dim=None,
        layernormalize=None,
    ):
        super(InverseDynamics, self).__init__(
            scope="inverse_dynamics",
            policy=policy,
            features_shared_with_policy=features_shared_with_policy,
            device=device,
            feature_dim=feature_dim,
            layernormalize=layernormalize,
        )
        """Fully connected layer taking the extracted features for the current state and
        the next state as input (model specified in FeatureExtractor), applying relu for
        the hidden activations, followed by another fc layer outputting the predicted
        action."""
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim * 2, self.policy.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.policy.hidden_dim, self.ac_space.n),
        ).to(self.device)
        # Add auxiliary task output layers to list of optimized parameters
        self.param_list.extend(self.fc.parameters())
        # Initialize weights with xavier_uniform and constant biases
        self.init_weight()

    def init_weight(self):
        for m in self.fc:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def get_loss(self):
        """Calculate the auxiliary loss (backward loss). This is the cross entropy
        between the predicted action probabilities and the actions actually performed.

        Returns
        -------
        tensor
            Losses for each action prediction.

        """
        x = torch.cat([self.features, self.next_features], 2)
        x = flatten_dims(x, 1)
        # Get action probabilities for each action. shape=[nsteps_per_seg, act_dim]
        param = self.fc(x)
        # Create probability distribution from logits.
        idfpd = self.policy.ac_pdtype.pdfromflat(param)
        # Get the actions that were actually performed and flatten.
        # shape=[n_steps_per_seg]
        ac = flatten_dims(self.ac, len(self.ac_space.shape))
        # Calculate the cross entropy between the logits of the action predictions and
        # the actual actions. shape=[n_steps_per_seg, 1]
        return idfpd.neglogp(ac)


class VAE(FeatureExtractor):
    """Variational Autoencoder. Used to provide an auxiliary task for learning the
    weights of the FeatureExtractor.

    Parameters
    ----------
    policy : CnnPolicy
        The policy of the ppo agent.
    features_shared_with_policy : bool
        Whether to use the policy features for the disagreement module.
    feature_dim : int
        Number of neurons in the last layer of the feature network (or middle of VAE).
    layernormalize : bool
        Whether to normalize output of the last layer.
    device: torch.device
        Which device to optimize the model on.
    spherical_obs : bool
        Whether to scale the latent space to be spherical.

    Attributes
    ----------
    features_model : torch.nn.Module
        Convolutional network for feature extraction from observations. Feature dim is
        twice as big as for other feature models but only half is used for dynamics.
    ob_space : Space
        Observation space properties (from env.observation_space).
    decoder_model : torch.nn.Module
        Deconvolutional network for reconstruction.
    param_list : list
        List of parameters to be optimized.
    features_std : type
        Second half of features. Used for posterior scale.
    scale : torch.nn.Parameter
        Scaling parameter when sperical_obs=True.

    """

    def __init__(
        self,
        policy,
        features_shared_with_policy,
        device,
        feature_dim=None,
        layernormalize=False,
        spherical_obs=False,
    ):
        assert (
            not layernormalize
        ), "VAE features should already have reasonable size, no need to normalize them"
        super(VAE, self).__init__(
            scope="vae",
            policy=policy,
            features_shared_with_policy=features_shared_with_policy,
            device=device,
            feature_dim=feature_dim,
            layernormalize=False,
        )
        # Initialize the feature model. Note that the feature dim is twice as big.
        self.features_model = small_convnet(
            self.ob_space,
            nonlinear=torch.nn.LeakyReLU,
            feature_dim=2 * self.feature_dim,
            last_nonlinear=None,
            layernormalize=False,
            device=self.device,
        ).to(self.device)
        # Initialize the decoder model.
        self.decoder_model = small_deconvnet(
            self.ob_space,
            feature_dim=self.feature_dim,
            nonlinear=torch.nn.LeakyReLU,
            ch=4 if spherical_obs else 8,
            positional_bias=True,
            device=self.device,
        ).to(self.device)

        # Add encoder and decoder to optimized parameters.
        self.param_list = []  # remove smaller feature model that was replaced
        self.param_list.extend(self.features_model.parameters())
        self.param_list.extend(self.decoder_model.parameters())

        self.features_std = None
        self.spherical_obs = spherical_obs
        if self.spherical_obs:
            # Initialize a scale paramtere and add it to optimized parameters.
            # TODO: verify - is the scale really a param that requires grad?
            self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.param_list.extend([self.scale])

    def update_features(self, obs, last_obs):
        """Get the features corresponding to observations from the model and update.

        Parameters
        ----------
        obs : array
            Current observations.
        last_obs : array
            Last observations.

        """
        # Get features from the features model.
        features = self.get_features(obs)
        last_features = self.get_features(last_obs)
        # concatenate last and current features
        next_features = torch.cat([features[:, 1:], last_features], 1)
        # Split the features in two (remember, it was 2* feature_dim). The second part
        # is just used for the posterior scale calculation -> not needed for next feats
        self.features, self.features_std = torch.split(
            features, [self.feature_dim, self.feature_dim], -1
        )  # use means only for features exposed to dynamics
        self.next_features, _ = torch.split(
            next_features, [self.feature_dim, self.feature_dim], -1
        )
        # get current actions and observations
        self.ac = self.policy.ac
        self.ob = self.policy.ob

    def get_loss(self):
        """Calculate the auxiliary loss for the variational autoencoder task.

        Returns
        -------
        tensor
            Losses of the VAE.

        """
        posterior_mean = self.features
        # The softplus fuction is a smoothed version of the ReLu function.
        posterior_scale = torch.nn.functional.softplus(self.features_std)
        # Create a gaussian distribution with the first part of the features as mean and
        # the second as scale.
        posterior_distribution = torch.distributions.normal.Normal(
            posterior_mean, posterior_scale
        )

        sh = posterior_mean.shape
        # Get a prior distribution (gaussian) centered around 0 with std 1.
        prior = torch.distributions.normal.Normal(torch.zeros(sh), torch.ones(sh))
        # Calculate the KL divergence between posterior and prior and take the sum.
        posterior_kl = torch.distributions.kl.kl_divergence(
            posterior_distribution, prior
        )
        posterior_kl = torch.sum(posterior_kl, -1)
        assert len(posterior_kl.shape) == 2

        # Sample from the posterior distribution
        posterior_sample = (
            posterior_distribution.rsample()
        )  # do we need to let the gradient pass through the sample process?
        # Apply the decoder to the samples from the posterior distribution
        reconstruction_distribution = self.decoder(posterior_sample)

        # normalize, add noise and transpose the observations
        norm_obs = self.add_noise_and_normalize(self.ob)
        norm_obs = np.transpose(
            norm_obs, [i for i in range(len(norm_obs.shape) - 3)] + [-1, -3, -2]
        )

        # Get the log probability of the normed observations under the reconstruction
        # distribution and sum up this reconstruction likelihood.
        reconstruction_likelihood = reconstruction_distribution.log_prob(
            norm_obs.float()
        )
        assert reconstruction_likelihood.shape[-3:] == (4, 84, 84)
        reconstruction_likelihood = torch.sum(reconstruction_likelihood, [-3, -2, -1])

        # Calculate the difference between the overall reconstruction likelihood and the
        # posterior KL divergence. Return the negative of this as loss.
        likelihood_lower_bound = reconstruction_likelihood - posterior_kl
        return -likelihood_lower_bound

    def add_noise_and_normalize(self, x):
        # Add randomly distributed noise.
        x = x + np.random.normal(0.0, 1.0, size=x.shape)  # no bias
        # normalize the observations.
        x = (x - self.ob_mean) / self.ob_std
        return x

    def decoder(self, z):
        """Run latent space activations through the decoder model, apply spherical
        scaling if needed and get the distribution of the reconstructions.

        Parameters
        ----------
        z : Tensor
            Latent activations in the VAE after processing in the encoder.

        Returns
        -------
        tensor
            Reconstruction distribution.

        """
        z_has_timesteps = len(z.shape) == 3
        if z_has_timesteps:
            sh = z.shape
            z = flatten_dims(z, 1)
        # Run the latent vector trhough the decoder model
        z = self.decoder_model(z)
        # reshape if needed
        if z_has_timesteps:
            z = unflatten_first_dim(z, sh)

        # Calculate the scale parameter
        if self.spherical_obs:
            scale = torch.max(self.scale, torch.tensor(-4.0))
            scale = torch.nn.functional.softplus(scale)
            scale = scale * torch.ones(z.shape)
        else:
            z, scale = torch.split(z, [4, 4], -3)
            scale = torch.nn.functional.softplus(scale)
        # Return the scaled distribution of the decoder reconstruction.
        return torch.distributions.normal.Normal(z, scale)
