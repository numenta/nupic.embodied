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

import torch

from nupic.embodied.models import DynamicsNet
from nupic.embodied.utils.model_parts import flatten_dims, unflatten_first_dim


class Dynamics(object):
    """Dynamics module with feature extractor (can have auxiliary task). One feature
    extractor is used for n dynamics models. Disagreement between dynamics models is
    used for learning.

    Parameters
    ----------
    auxiliary_task : FeatureExtractor
        Feature extractor used to get and learn features.
    feature_dim : int
        Number of neurons in the feature network layers.
    scope : str
        Scope name of the dynamics model.

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
    param_list : type
        List of parameters to be optimized.
    features_model : model
        Feature model to use for feature extraction.
    ob_space : Space
        Observation space properties.
    dynamics_net : torch.nn.Module
        Network for dynamics task.
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
        hidden_dim,
        ac_space,
        ob_mean,
        ob_std,
        device,
        feature_dim=None,
        scope="dynamics",
    ):
        self.scope = scope
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.ac_space = ac_space
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        self.param_list = []
        self.features_model = None
        self.device = device

        # Initialize the loss network.
        self.dynamics_net = DynamicsNet(
            nblocks=4,
            feature_dim=self.feature_dim,
            ac_dim=self.ac_space.n,
            out_feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        # Add parameters of loss net to optimized parameters
        self.param_list.extend(self.dynamics_net.parameters())

    def get_predictions(self, ac, features):
        """Get the current prediction of the dynamics model.

        Returns
        -------
        array
            Returns the output of the dynamics network
        """
        # TODO: refactor this function, too many shape transformations in ac, confusing
        sh = ac.shape  # = [1, nsteps_per_seg]
        ac = flatten_dims(ac, len(self.ac_space.shape))  # shape = [nsteps_per_seg]
        # Turn actions into one hot encoding
        ac = torch.zeros(ac.shape + (self.ac_space.n,)).scatter_(
            1, ac.unsqueeze(1).type(torch.int64), 1
        )  # shape = [nsteps_per_seg, ac_space.n]

        sh = features.shape  # [1, nsteps_per_seg, feature_dim]
        x = flatten_dims(features, 1)  # [nsteps_per_seg, feature_dim]
        assert x.shape[:-1] == ac.shape[:-1]

        # forward pass of actions and features in dynamics net
        x = self.dynamics_net(x, ac)

        # reshape
        x = unflatten_first_dim(x, sh)  # [1, nsteps_per_seg, feature_dim]

        return x

    def get_prediction_error(self, predicted_state, next_features):
        """Get the current prediction of the dynamics model.

        Returns
        -------
        array
            Returns the mean squared difference between the output and next features.

        """
        return torch.mean((predicted_state - next_features) ** 2, -1)

    def get_predictions_partial(self, ac, features, next_features):
        """Get the loss of the dynamics model with dropout. The dynamics models are trained
        using the prediction error. The disagreement is only used as a reward signal for
        the policy. Dropout is added to the loss to enforce some variance between models
        while still using all of the data.

        Returns
        -------
        array
            Mean squared difference between the output and the next features.

        """

        x = self.get_predictions(ac, features)

        # Take the mean-squared diff between out features (input was current
        # features and action) and next features (shape=[1, nsteps_per_seg])
        loss = torch.mean((x - next_features) ** 2, -1)  # mean over frames
        # Apply dropout here to ensure variability between dynamics models. This is done
        # instead of bootstrapping the samples so that all samples can be used to train
        # every model.

        do = torch.nn.Dropout(p=0.2)
        do_loss = do(loss)
        return do_loss  # vector with mse for each feature

    # def predict_features(self, auxiliary_task):
    #     """
    #     Deprecated: remove it
    #     Forward pass of the dynamics model

    #     Parameters
    #     ----------
    #     obs : array
    #         batch of observations. shape = [n_env, nsteps_per_seg, W, H, C].
    #     last_obs : array
    #         batch of last observations. shape = [n_env, 1, W, H, C].
    #     acs : array
    #         batch of actions. shape = [n_env, nsteps_per_seg]

    #     Returns
    #     -------
    #     array
    #         predictions. shape = [n_env, nsteps_per_seg, feature_dim]

    #     """
    #     n_chunks = 8  # TODO: make this a hyperparameter?
    #     n = obs.shape[0]
    #     chunk_size = n // n_chunks
    #     assert n % n_chunks == 0

    #     def get_slice(i):
    #         """Get slice number i of chunksize n/n_chunks. So eg if we have 64 envs
    #         and 8 chunks then the chunksize is 8 and the first slice is 0:8,the second
    #         8:16, the third 16:24, ...

    #         Parameters
    #         ----------
    #         i : int
    #             slice number.

    #         Returns
    #         -------
    #         slice
    #             slice to specify which part of teh observations to process.

    #         """
    #         return slice(i * chunk_size, (i + 1) * chunk_size)

    #     predictions = []

    #     for i in range(n_chunks):
    #         # process the current slice of observations
    #         ob = obs[get_slice(i)]
    #         last_ob = last_obs[get_slice(i)]
    #         ac = acs[get_slice(i)]
    #         # update the policy features and the auxiliary task features
    #         # self.auxiliary_task.policy.update_features(ob, ac)
    #         # self.auxiliary_task.update_features(ob, last_ob)
    #         # get the updated features
    #         self.update_features(auxiliary_task)

    #         # get the prediction from the model corresponding with the new features
    #         predictions.append(self.get_predictions())

    #     predictions = torch.cat(predictions, 0)

    #     return predictions
