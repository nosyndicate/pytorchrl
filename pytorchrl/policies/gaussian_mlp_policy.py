import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from rllab.misc.overrides import overrides

from pytorchrl.core.parameterized import Parameterized
from pytorchrl.distributions.diagonal_gaussian import DiagonalGaussian
from pytorchrl.policies.base import StochasticPolicy


class GaussianMLPPolicy(StochasticPolicy, Parameterized):
    """
    Stochastic policy as Gaussian distribution.
    """
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes=(32, 32),
        adaptive_std=True,
        std_based_on_state=False,
        init_std=1.0,
        std_share_network=False,
        std_hidden_size=(32, 32),
        min_std=1e-6,
        hidden_nonlinearity=nn.Tanh,
        std_hidden_nonlinearity=nn.Tanh,
        output_nonlinearity=None,
        dist_cls=DiagonalGaussian,
    ):
        """
        Create Policy Network

        Parameters
        ----------
        adaptive_std (bool): Whether to change std of the network, if false,
            std will always be init_std.
        std_based_on_state (bool): Whether std is a function of state,
            std = func(state), or just a array of numbers. This option is
            only effective is adaptive_std is True.
        init_std (float): Initial std. This only effective if
            std_based_on_state if False.
        std_share_network (bool): Whether share the network parameter,
            if this is true, the mean and std is only separate at the
            last layer. Otherwise, std and mean have two different network.
        """
        super(GaussianMLPPolicy, self).__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.adaptive_std = adaptive_std
        self.std_share_network = std_share_network
        self.init_std = init_std
        self.std_based_on_state = std_based_on_state
        self.min_std = min_std

        sizes = [self.observation_dim] + list(hidden_sizes)
        std_hidden_size = [self.observation_dim] + list(std_hidden_size)
        # Create the tensor has the same shape as action
        init_std_tensor = torch.Tensor(self.action_dim).fill_(init_std)

        if adaptive_std:
            if self.std_based_on_state:
                if self.std_share_netowrk:
                    self.model, self.mean_model, self.log_std_model = \
                        self.define_shared_network(sizes,
                            hidden_nonlinearity, output_nonlinearity)
                else:
                    self.mean_model = self.define_single_network(sizes,
                        hidden_nonlinearity, output_nonlinearity)
                    self.log_std_model = self.define_single_network(
                        std_hidden_size, std_hidden_nonlinearity,
                        output_nonlinearity=None)
            else:
                self.mean_model = self.define_single_network(sizes,
                        hidden_nonlinearity, output_nonlinearity)
                self.log_std_model = nn.Parameter(init_std_tensor)
        else:
            self.mean_model = self.define_single_network(sizes,
                hidden_nonlinearity, output_nonlinearity)
            self.log_std_model = Variable(init_std_tensor)

        self.dist_cls = dist_cls


    def define_shared_network(self, sizes, hidden_nonlinearity,
        output_nonlinearity):
        """
        Define the network output both mean and log_std of the
        policy with shared parameters.

        Parameters
        ----------
        sizes (list): A list of integer specify the size of each layer
            from input to last hidden layer.
        output_nonlinearity: #TODO (ewei)

        Returns
        -------
        model (nn.Module): A Module take input and output the Variable
            of last hidden layer
        mean (nn.Module): A layer take the last hidden layer output
            and produce the mean of the policy
        log_std (nn.Module): A layer take the last hidden layer output
            and produce the log std of the policy.
        """
        # Create the base model
        submodules = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                submodules.append(nn.Linear(size, sizes[index + 1]))
                submodules.append(hidden_nonlinearity())

        model = nn.Sequential(*submodules)

        # Create the layer for means
        mean_modules = []
        last_hidden_size = sizes[len(sizes) - 1]
        mean_modules.append(nn.Linear(last_hidden_size, self.action_dim))
        if output_nonlinearity is not None:
            mean_modules.append(output_nonlinearity())
        means = nn.Sequential(*mean_modules)

        # Create the layer for log stds
        log_stds = nn.Linear(last_hidden_size, self.action_dim)

        return model, means, log_stds

    def define_single_network(self, sizes, hidden_nonlinearity,
        output_nonlinearity):
        """
        Define a single input - single output network (thus, no shared
        part).

        Parameters
        ----------
        sizes (list):
        hidden_nonlinearity ():
        output_nonlinearity ():
        """
        submodules = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                submodules.append(nn.Linear(size, sizes[index + 1]))
                submodules.append(hidden_nonlinearity())

        last_hidden_size = sizes[len(sizes) - 1]
        submodules.append(nn.Linear(last_hidden_size, self.action_dim))
        if output_nonlinearity is not None:
            submodules.append(output_nonlinearity())
        model = nn.Sequential(*submodules)

        return model

    def forward(self, observations):
        """
        Return action for given state

        Parameters
        ----------
        observations (Variable): state wrapped in Variable

        Returns
        -------
        means (Variable): mean of the gaussian policy
        log_stds (Variable): log of std of the gaussian policy
        """
        if self.adaptive_std:
            if self.std_based_on_state:
                if self.std_share_netowrk:
                    last_hidden = self.model(observations)
                    means = self.mean_model(last_hidden)
                    log_stds = self.log_std_model(last_hidden)
                else:
                    means = self.mean_model(observations)
                    log_stds = self.log_std_model(observations)
            else:
                means = self.mean_model(observations)
                # Expand the log_stds
                log_stds = self.log_std_model.expand_as(means)
        else:
            means = self.mean_model(observations)
            # Expand the log_stds
            log_stds = self.log_std_model.expand_as(means)

        # Set the minimum log_std
        if self.min_std is not None:
            min_log_std = Variable(torch.Tensor(
                np.log([self.min_std]))).type(torch.FloatTensor)
            # max operation requires two variables
            log_stds = torch.max(log_stds, min_log_std)

        return means, log_stds

    @overrides
    def get_action(self, observation):
        """
        Get observation and return the action for the observation.

        Parameters
        ----------
        observation (numpy.ndarray): current observation.

        Return
        ------
        action (numpy.ndarray): action for the observation.
        agent_info (dict): Additional info for the agent
        """
        # Need to first convert observation into Variable
        # We do not need to call backward in get_action method
        # Thus, we set volatile to True.
        obs_variable = Variable(torch.from_numpy(observation),
            volatile=True).type(torch.FloatTensor)

        means_variable, log_stds_variable = self.forward(obs_variable)
        means = means_variable.data.numpy()
        log_stds = log_stds_variable.data.numpy()
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means

        return actions, dict(mean=means, log_std=log_stds)

    def get_actions(self, observations):
        """
        Get observation and returnt the actions for the observations.
        This is the batch version of the get_action method.

        If the input observations is of size n * obs_dim, where
        n is the batch size, and obs_dim is the dimension of the observation,
        then the actions, means, and log_stds in the return values is
        of size n * act_dim, where act_dim is the size of action dimension.

        Parameters
        ----------
        observations (numpy.ndarray): current observation.

        Returns
        -------
        actions (numpy.ndarray): actions for the observations.
        agent_infos (dict): Additional info for the agent
        """
        obs_variable = Variable(torch.from_numpy(observations),
            volatile=True).type(torch.FloatTensor)

        means_variable, log_stds_variable = self.forward(obs_variable)

        means = means_variable.data.numpy()
        log_stds = log_stds_variable.data.numpy()

        rnd = np.random.normal(size=means.shape)

        actions = rnd * np.exp(log_stds) + means

        return actions, dict(mean=means, log_std=log_stds)

    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()

    def get_policy_distribution(self, obs_var):
        """
        Return the distribution of this policy based on observation.
        This is different from the method 'distribution' in that, we
        need to do a forward pass to get mean and log_stds using the
        neural network.

        Parameters
        ----------
        obs_var (Variable):

        Returns
        -------
        distribution (DiagonalGaussian):
        """
        means_variable, log_stds_variable = self.forward(obs_var)
        return self.dist_cls.from_dict(means_variable, log_stds_variable)

    def distribution(self, dist_info):
        """
        Return the distribution of this policy based on the provided info.
        No forward pass is needed.

        Parameters
        ----------
        dist_info (dict):

        Returns
        -------
        distribution (DiagonalGaussian):
        """
        means = dist_info['mean']
        log_stds = dist_info['log_std']
        if isinstance(means, np.ndarray):
            means = Variable(torch.from_numpy(means)).type(torch.FloatTensor)
        if isinstance(log_stds, np.ndarray):
            log_stds = Variable(torch.from_numpy(log_stds)).type(torch.FloatTensor)

        return self.dist_cls.from_dict(means, log_stds)