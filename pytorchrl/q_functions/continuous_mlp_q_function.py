import torch
from torch.autograd import Variable
import torch.nn as nn

from rllab.misc.overrides import overrides

from pytorchrl.core.parameterized import Parameterized


class ContinuousMLPQFunction(nn.Module, Parameterized):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.ReLU
    ):
        """
        Create the Q-network.

        Parameters
        ----------
        observation_dim (int): dimensionality of observation space
        action_dim (int): dimensionality of action space.
        hidden_sizes (tuple): sizes of hidden layer, every two successive
            integer define a hidden layer.
        hidden_nonlinearity (nn.Module): Module for activation function.
        """
        super(ContinuousMLPQFunction, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Define network
        layers = []
        sizes = [int(self.observation_dim + self.action_dim)] + list(hidden_sizes)
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                layers.append(nn.Linear(size, sizes[index + 1]))
                layers.append(hidden_nonlinearity())
        # Add the last layer
        layers.append(nn.Linear(sizes[len(sizes) - 1], 1))
        self.model = nn.Sequential(*layers)

        # NOTE : We didn't use He's initialzation for previous layer
        # This is different from the DDPG paper.
        for param in self.model.parameters():
            nn.init.uniform(param, -3e-3, 3e-3)

    def forward(self, obs_variable, actions_variable):
        """
        Return Q(s, a) for given state and action

        Parameters
        ----------
        obs_variable (Variable): state wrapped in Variable
        actions_variable (Variable): action wrapped in Variable

        Returns
        -------
        q_val_variable (Variable): Q value of the state and action,
            wrapped in Variable
        """
        state_action = torch.cat((obs_variable, actions_variable), 1)
        q_val_variable = self.model(state_action)

        return q_val_variable

    def get_qval(self, observation, action):
        """
        Return Q(s, a) for given state and action. This is
        used for determine the Q(s', a') on target network,
        thus, we do not need to call backward on the model.
        So we set Variables to violatile True.

        Parameters
        ----------
        observation (numpy.ndarray): state
        action (numpy.ndarray): action

        Returns
        -------
        q_val (numpy.ndarray): Q value of the state and action
        """
        obs_variable = Variable(torch.from_numpy(observation),
            volatile=True).type(torch.FloatTensor)

        # obs_variable sets volatile to True, thus, we do not set
        # it here
        action_variable = Variable(torch.from_numpy(action)).type(
            torch.FloatTensor)

        q_value_variable = self.forward(obs_variable, action_variable)
        return q_value_variable.data.numpy()

    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()
