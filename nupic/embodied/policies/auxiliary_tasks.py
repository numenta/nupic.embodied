import numpy as np
import torch
from nupic.embodied.utils.model_parts import (
    small_convnet,
    flatten_dims,
    unflatten_first_dim,
    small_deconvnet,
)


class FeatureExtractor(object):
    """Extract features from high dimensional observations.

    Parameters
    ----------
    policy : CnnPolicy
        The policy of the ppo agent.
    features_shared_with_policy : bool
        Whether to use the policy features for the disagreement module.
    feat_dim : int
        Number of neurons in the last layer of the feature network (or middle of VAE).
    layernormalize : bool
        Whether to normalize output of the last layer.
    scope : str
        torch model scope.
    device: torch.device
        Which device to optimize the model on.

    Attributes
    ----------
    hid_dim : int
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
    param_list : type
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
        feat_dim=None,
        layernormalize=None,
        scope="feature_extractor",
    ):
        self.scope = scope
        self.features_shared_with_policy = features_shared_with_policy
        self.feat_dim = feat_dim
        self.layernormalize = layernormalize
        self.policy = policy
        self.hid_dim = policy.hid_dim
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
                feat_dim=self.feat_dim,
                last_nonlinear=None,
                layernormalize=self.layernormalize,
                device=self.device,
            ).to(self.device)
            # Add feature model to optimization parameters
            self.param_list = self.param_list + [
                dict(params=self.features_model.parameters())
            ]

        self.scope = scope

        self.features = None
        self.next_features = None
        self.ac = None
        self.ob = None

    def update_features(self, obs, last_obs):
        """Get features from feature model and set self.features and self.next_features.
        Also sets self.ac to the last actions at end of rollout..

        Parameters
        ----------
        obs : array
            Current observations.
        last_obs : array
            Previous observations.

        """
        if self.features_shared_with_policy:
            # Get features corresponding with the observation from the policy network
            self.features = self.policy.flat_features
            last_features = self.policy.get_features(last_obs)
        else:
            # Get features corresponding with the observation from the feature network
            self.features = self.get_features(obs)
            last_features = self.get_features(last_obs)
        # concatenate the last and current features
        self.next_features = torch.cat([self.features[:, 1:], last_features], 1)

        self.ac = self.policy.ac
        self.ob = self.policy.ob

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
        has_timesteps = len(obs.shape) == 5
        if has_timesteps:
            sh = obs.shape
            obs = flatten_dims(obs, len(self.ob_space.shape))
        # Normalize observations
        obs = (obs - self.ob_mean) / self.ob_std
        # Reshape observations
        obs = np.transpose(obs, [i for i in range(len(obs.shape) - 3)] + [-1, -3, -2])
        # Get features from the features_model
        act = self.features_model(torch.tensor(obs).to(self.device))

        if has_timesteps:
            act = unflatten_first_dim(act, sh)

        """ print(
            "features: "
            + str(act.shape)
            + " mean: "
            + str(np.mean(act.data.numpy()))
            + " var: "
            + str(np.var(act.data.numpy()))
        )"""
        return act

    def get_loss(self, *arg, **kwarg):
        # is 0 because we use no auxiliary task for feature learning
        return torch.tensor(0.0).to(self.device)


class InverseDynamics(FeatureExtractor):
    # TODO: Add comments for rest of script
    # TODO: Add .to(self.device)
    def __init__(
        self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None
    ):
        super(InverseDynamics, self).__init__(
            scope="inverse_dynamics",
            policy=policy,
            features_shared_with_policy=features_shared_with_policy,
            feat_dim=feat_dim,
            layernormalize=layernormalize,
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.feat_dim * 2, self.policy.hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.policy.hid_dim, self.ac_space.n),
        )
        self.param_list = self.param_list + [dict(params=self.fc.parameters())]
        self.init_weight()

    def init_weight(self):
        for m in self.fc:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def get_loss(self):
        x = torch.cat([self.features, self.next_features], 2)
        # sh = x.shape
        x = flatten_dims(x, 1)
        param = self.fc(x)
        idfpd = self.policy.ac_pdtype.pdfromflat(param)
        ac = flatten_dims(self.ac, len(self.ac_space.shape))
        return idfpd.neglogp(torch.tensor(ac))


class VAE(FeatureExtractor):
    def __init__(
        self,
        policy,
        features_shared_with_policy,
        feat_dim=None,
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
            feat_dim=feat_dim,
            layernormalize=False,
        )

        self.features_model = small_convnet(
            self.ob_space,
            nonlinear=torch.nn.LeakyReLU,
            feat_dim=2 * self.feat_dim,
            last_nonlinear=None,
            layernormalize=False,
        )
        self.decoder_model = small_deconvnet(
            self.ob_space,
            feat_dim=self.feat_dim,
            nonlinear=torch.nn.LeakyReLU,
            ch=4 if spherical_obs else 8,
            positional_bias=True,
        )

        self.param_list = [
            dict(params=self.features_model.parameters()),
            dict(params=self.decoder_model.parameters()),
        ]

        self.features_std = None
        self.next_features_std = None

        self.spherical_obs = spherical_obs
        if self.spherical_obs:
            self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.param_list = self.param_list + [dict(params=[self.scale])]

    def update_features(self, obs, last_obs):
        features = self.get_features(obs)
        last_features = self.get_features(last_obs)
        next_features = torch.cat([features[:, 1:], last_features], 1)

        self.features, self.features_std = torch.split(
            features, [self.feat_dim, self.feat_dim], -1
        )  # use means only for features exposed to dynamics
        self.next_features, self.next_features_std = torch.split(
            next_features, [self.feat_dim, self.feat_dim], -1
        )
        self.ac = self.policy.ac
        self.ob = self.policy.ob

    def get_loss(self):
        posterior_mean = self.features
        posterior_scale = torch.nn.functional.softplus(self.features_std)
        posterior_distribution = torch.distributions.normal.Normal(
            posterior_mean, posterior_scale
        )

        sh = posterior_mean.shape
        prior = torch.distributions.normal.Normal(torch.zeros(sh), torch.ones(sh))

        posterior_kl = torch.distributions.kl.kl_divergence(
            posterior_distribution, prior
        )

        posterior_kl = torch.sum(posterior_kl, -1)
        assert len(posterior_kl.shape) == 2

        posterior_sample = (
            posterior_distribution.rsample()
        )  # do we need to let the gradient pass through the sample process?
        reconstruction_distribution = self.decoder(posterior_sample)
        norm_obs = self.add_noise_and_normalize(self.ob)
        norm_obs = np.transpose(
            norm_obs, [i for i in range(len(norm_obs.shape) - 3)] + [-1, -3, -2]
        )  # transpose channel axis
        reconstruction_likelihood = reconstruction_distribution.log_prob(
            torch.tensor(norm_obs).float()
        )
        assert reconstruction_likelihood.shape[-3:] == (4, 84, 84)
        reconstruction_likelihood = torch.sum(reconstruction_likelihood, [-3, -2, -1])

        likelihood_lower_bound = reconstruction_likelihood - posterior_kl
        return -likelihood_lower_bound

    def add_noise_and_normalize(self, x):
        x = x + np.random.normal(0.0, 1.0, size=x.shape)  # no bias
        x = (x - self.ob_mean) / self.ob_std
        return x

    def decoder(self, z):
        z_has_timesteps = len(z.shape) == 3
        if z_has_timesteps:
            sh = z.shape
            z = flatten_dims(z, 1)
        z = self.decoder_model(z)
        if z_has_timesteps:
            z = unflatten_first_dim(z, sh)
        if self.spherical_obs:
            scale = torch.max(self.scale, torch.tensor(-4.0))
            scale = torch.nn.functional.softplus(scale)
            scale = scale * torch.ones(z.shape)
        else:
            z, scale = torch.split(z, [4, 4], -3)
            scale = torch.nn.functional.softplus(scale)
        return torch.distributions.normal.Normal(z, scale)
