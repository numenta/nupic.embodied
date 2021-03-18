import torch

from nupic.embodied.utils.model_parts import flatten_dims, unflatten_first_dim


class Dynamics(object):
    """Dynamics module with feature extractor (can have auxiliary task). One feature
    extractor is used for n dynamics models. Disagreement between dynamics models is
    used for learning.

    Parameters
    ----------
    auxiliary_task : FeatureExtractor
        Feature extractor used to get and learn features.
    var_output : bool
        If loss should be calculated from the variance over dynamics model features.
        If false, the loss is the variance over the error of state features and next
        state features between the different dynamics models.
    feat_dim : int
        Number of neurons in the feature network layers.
    scope : str
        Scope name of the dynamics model.

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
        auxiliary_task,
        var_output,
        feat_dim=None,
        scope="dynamics",
    ):
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hid_dim = self.auxiliary_task.hid_dim
        self.feat_dim = feat_dim
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        self.param_list = []
        self.var_output = var_output
        self.features_model = None

        # Initialize the loss network.
        self.dynamics_net = dynamics_net(
            nblocks=4,
            feat_dim=self.feat_dim,
            ac_dim=self.ac_space.n,
            out_feat_dim=self.feat_dim,
            hid_dim=self.hid_dim,
        )
        # Add parameters of loss net to optimized parameters
        self.param_list = self.param_list + [
            dict(params=self.dynamics_net.parameters())
        ]

        self.features = None
        self.next_features = None
        self.ac = None
        self.ob = None

    def update_features(self):
        """Get features from the feature network."""
        # get the features with detach -> no gradient will go from here
        self.features = self.auxiliary_task.features.detach()
        self.next_features = self.auxiliary_task.next_features.detach()
        self.ac = self.auxiliary_task.ac
        self.ob = self.auxiliary_task.ob

    def get_loss(self):
        """Get the current loss of the dynamics model.

        Returns
        -------
        array
            If var_output=True returns the output of the loss network, otherwise the
            mean squared difference between the output and the next features.

        """
        ac = self.ac
        sh = ac.shape  # = [1, nsteps_per_seg]
        ac = flatten_dims(ac, len(self.ac_space.shape))  # shape = [nsteps_per_seg]
        # Turn actions into one hot encoding
        ac = torch.zeros(ac.shape + (self.ac_space.n,)).scatter_(
            1, torch.tensor(ac).unsqueeze(1), 1
        )  # shape = [nsteps_per_seg, ac_space.n]

        features = self.features
        sh = features.shape  # [1, nsteps_per_seg, feature_dim]
        x = flatten_dims(features, 1)  # [nsteps_per_seg, feature_dim]
        assert x.shape[:-1] == ac.shape[:-1]

        # forward pass of actions and features in dynamics net
        x = self.dynamics_net(x, ac)  # [nsteps_per_seg, feature_dim]

        # reshape
        x = unflatten_first_dim(x, sh)  # [1, nsteps_per_seg, feature_dim]
        if self.var_output:
            # Return output from dynamics network
            # (shape=[1, nsteps_per_seg, next_feat_dim])
            return x
        else:
            # Take the mean-squared diff between out features (input was current
            # features and action) and next features (shape=[1, nsteps_per_seg])
            next_features = self.next_features
            return torch.mean((x - next_features) ** 2, -1)

    def get_loss_partial(self):
        """Get the loss of the dynamics model with dropout. No var_output option..

        Returns
        -------
        array
            Mean squared difference between the output and the next features.

        """
        ac = self.ac
        sh = ac.shape  # = [1, nsteps_per_seg]
        ac = flatten_dims(ac, len(self.ac_space.shape))  # shape = [nsteps_per_seg]
        # Turn actions into one hot encoding
        ac = torch.zeros(ac.shape + (self.ac_space.n,)).scatter_(
            1, torch.tensor(ac).unsqueeze(1), 1
        )  # shape = [nsteps_per_seg, ac_space.n]

        features = self.features
        sh = features.shape  # [1, nsteps_per_seg, feature_dim]
        x = flatten_dims(features, 1)  # [nsteps_per_seg, feature_dim]
        assert x.shape[:-1] == ac.shape[:-1]

        # forward pass of actions and features in dynamics net
        x = self.dynamics_net(x, ac)  # [nsteps_per_seg, feature_dim]

        # reshape
        x = unflatten_first_dim(x, sh)  # [1, nsteps_per_seg, feature_dim]
        # Take the mean-squared diff between out features (input was current
        # features and action) and next features (shape=[1, nsteps_per_seg])
        next_features = self.next_features
        loss = torch.mean((x - next_features) ** 2, -1)
        # TODO: Why is dropout applied here?
        do = torch.nn.Dropout(p=0.2)
        do_loss = do(loss)
        return do_loss

    def calculate_loss(self, obs, last_obs, acs):
        """Short summary.

        Parameters
        ----------
        obs : array
            batch of observations. shape = [n_env, nsteps_per_seg, W, H, C].
        last_obs : array
            batch of last observations. shape = [n_env, 1, W, H, C].
        acs : array
            batch of actions. shape = [n_env, nsteps_per_seg]

        Returns
        -------
        array
            losses. shape = [n_env, nsteps_per_seg, feat_dim]

        """
        n_chunks = 8  # TODO: make this a hyperparameter?
        n = obs.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0

        def get_slice(i):
            """Get slice number i of chunksize n/n_chunks. So eg if we have 64 envs
            and 8 chunks then the chunksize is 8 and the first slice is 0:8, the second
            8:16, the third 16:24, ...

            Parameters
            ----------
            i : int
                slice number.

            Returns
            -------
            slice
                slice to specify which part of teh observations to process.

            """
            return slice(i * chunk_size, (i + 1) * chunk_size)

        losses = None

        for i in range(n_chunks):
            # process the current slice of observations
            ob = obs[get_slice(i)]
            last_ob = last_obs[get_slice(i)]
            ac = acs[get_slice(i)]
            # update the policy features and the auxiliary task features
            self.auxiliary_task.policy.update_features(ob, ac)
            self.auxiliary_task.update_features(ob, last_ob)
            # get the updated features
            self.update_features()
            # get the loss from the loss model corresponding with the new features
            loss = self.get_loss()
            if losses is None:
                losses = loss
            else:
                # concatenate the losses from the current slice
                losses = torch.cat((losses, loss), 0)
        return losses.data.numpy()


class dynamics_net(torch.nn.Module):
    """Residual network to get the dynamics loss using the features from the auxiliary
    task model.

    Parameters
    ----------
    nblocks : int
        Number of residual blocks in the dynamics network.
    feat_dim : int
        Number of features from the feature network.
    ac_dim : int
        Action dimensionality.
    out_feat_dim : int
        Number of features from the feature network for the next state (usually same).
    hid_dim : int
        Number of neurons in the hidden layers.
    activation : torch.nn
        Activation function.

    Attributes
    ----------
    model_list : list
        List of torch model elements.

    """

    def __init__(
        self,
        nblocks,
        feat_dim,
        ac_dim,
        out_feat_dim,
        hid_dim,
        activation=torch.nn.LeakyReLU,
    ):
        super(dynamics_net, self).__init__()
        self.nblocks = nblocks
        self.feat_dim = feat_dim
        self.ac_dim = ac_dim
        self.out_feat_dim = out_feat_dim
        self.activation = activation

        # First layer of the model takes state features + actions as input and outputs
        # hid_dim activations
        model_list = [torch.nn.Linear(feat_dim + ac_dim, hid_dim), activation()]
        # n residual blocks with two linear layers each, teh first layer uses a non-
        # linear activation function.
        for _ in range(self.nblocks):
            model_list.append(torch.nn.Linear(hid_dim + ac_dim, hid_dim))
            model_list.append(activation())
            model_list.append(torch.nn.Linear(hid_dim + ac_dim, hid_dim))
        # Last layer takes hidden activations + actions as input and outputs features of
        # the size of the next_state features
        model_list.append(torch.nn.Linear(hid_dim + ac_dim, out_feat_dim))
        self.model_list = model_list
        # initialize the weights with xavier uniform initialization
        self.init_weight()

    def init_weight(self):
        for m in self.model_list:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)  # aka glorot initializer
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, features, ac):
        """Get output of a forward pass through the dynamics network.

        Parameters
        ----------
        features : array
            Features from the auxiliary network corresponding with the current state.
        ac : array
            Current actions.

        Returns
        -------
        array
            Features of the residual dynamics moduel from the current states & actions

        """
        idx = 0
        # concatenate state features with actions
        x = torch.cat((features, ac), dim=-1)  # shape=[nsteps_per_seg, feat_dim +n_act]
        for _ in range(2):
            x = self.model_list[idx](x)  # shape = [nsteps_per_seg, hid_dim]
            idx += 1
        for _ in range(self.nblocks):
            x0 = x
            for _ in range(3):
                if isinstance(self.model_list[idx], torch.nn.Linear):
                    # shape = [nsteps_per_seg, feat_dim + n_act]
                    x = torch.cat((x, ac), dim=-1)
                # shape = [nsteps_per_seg, hid_dim]
                x = self.model_list[idx](x)
                idx += 1
            x = x + x0  # shape = [nsteps_per_seg, hid_dim]
        x = torch.cat((x, ac), dim=-1)  # shape = [nsteps_per_seg, feat_dim + n_act]
        x = self.model_list[idx](x)  # shape = [nsteps_per_seg, out_feat_dim]
        assert idx == len(self.model_list) - 1
        assert x.shape[-1] == self.out_feat_dim
        return x
