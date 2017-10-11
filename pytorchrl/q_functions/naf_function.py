import torch
from torch.autograd import Variable
import torch.nn as nn

from rllab.misc.overrides import overrides

from pytorchrl.core.parameterized import Parameterized

class NAFQFunction(Policy, Parameterized):
    def __init__(
        self,
        observation_dim,
        action_dim,
        vf_hidden_size=(32, 32),
        vf_nonlinearity=nn.ReLU,
        mean_hidden_size=(32, 32),
        mean_nonlinearity=nn.ReLU,
        covariance_hidden_size=(32, 32),
        covariance_nonlinearity=nn.ReLU,
    ):

        """
        Create the Normalizad Advantage Function

        Note, although NAF is conceptually Q-function, we can actually
        call get_action on it, thus, we set this as the subclass of
        Policy.

        Parameters
        ----------
        observation_dim (int): dimensionality of observation space
        action_dim (int): dimensionality of action space.
        vf_hidden_size (tuple): sizes of hidden layers of value fuction V,
             every two successive integer define a hidden layer.
        vf_nonlinearity (nn.Module): Module for activation function.
        mean_hidden_size (tuple): sizes of hidden layers of mean of advantage
            fuction A.
        mean_nonlinearity (nn.Module): activation function for the
            mean of the quadractic function
        covariance_hidden_size (tuple): sizes of hidden layers of covariance
            of advantage fuction A.
        covariance_nonlinearity (nn.Module): activation function for
            covariance of the quadratic function.

        """
        super(NAFQFunction, self).__init__()

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        # Define network
        self.v = self.define_single_network(vf_hidden_size, vf_nonlinearity, 1)
        self.mu = self.define_single_network(mean_hidden_size, mean_nonlinearity,
            self.action_dim, output_nonlinearity=nn.Tanh)
        self.l = self.define_single_network(
            covariance_hidden_size, covariance_nonlinearity,
            (self.action_dim * (self.action_dim + 1) / 2))

        # Initialization

    def define_single_network(sizes, nonlinearity, output_size,
        output_nonlinearity=None):
        """
        """
        sizes = [self.observation_dim] + list(sizes)
        layers = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                layers.append(nn.Linear(size, sizes[index + 1]))
                layers.append(hidden_nonlinearity())

        # Add last layer
        layers.append(nn.Linear(sizes[len(sizes) - 1], output_size))
        return nn.Sequential(*layers)

    def forward(self, obs_var, actions_var):
        v = self.forward_v(obs_var)
        mu = self.forward_mu(obs_var)
        l = self.forward_l(obs_var)


    def forward_v(self, obs_variable):
        pass

    def forward_mu(self, obs_variable):
        """
        Return the mu value (mean action) for the observation.
        """
        return self.mu(obs_variable)

    def forward_l(self, obs_variable):
        pass


    def get_action(self, observation):
        """
        Get observation and return the action for the observation.

        Parameters
        ----------
        observation (numpy.ndarray): current observation.

        Return
        ------
        action (numpy.ndarray): action for the observation.
        other_info (dict): Additional info for the action
        """
        obs_variable = Variable(torch.from_numpy(observation).type(
            torch.FloatTensor), volatile=True)

        action = self.forward_mu(obs_variable)

        return action.data.numpy(), dict()