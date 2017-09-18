import torch
from torch.autograd import Variable
import torch.nn as nn

from rllab.misc.overrides import overrides

from pytorchrl.policies.base import Policy
from pytorchrl.core.parameterized import Parameterized


class DeterministicMLPPolicy(Policy, Parameterized):
    """
    Deterministic Policy
    """
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.ReLU
    ):
        """
        Create Policy Network.

        Parameters
        ----------
        observation_dim (int): observation dimension. If this is not a
            integer, than it should be compatible with type integer.
        action_dim (int): action dimension. If this is not a integer, than
            it should be compatible with type integer.
        # TODO (ewei)
        """
        super(DeterministicMLPPolicy, self).__init__()

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        # Define network
        sizes = [self.observation_dim] + list(hidden_sizes)
        submodules = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                submodules.append(nn.Linear(size, sizes[index + 1]))
                submodules.append(hidden_nonlinearity())

        # Add the last layer
        submodules.append(nn.Linear(sizes[len(sizes) - 1], self.action_dim))
        submodules.append(nn.Tanh())

        self.model = nn.Sequential(*submodules)

        # NOTE : We didn't use He's initialzation for previous layer
        for param in self.model.parameters():
            nn.init.uniform(param, -3e-3, 3e-3)

    def forward(self, observations):
        """
        Return action for given state

        Parameters
        ----------
        observations (Variable): state wrapped in Variable

        Returns
        -------
        action (Variable): action wrapped in Variable
        """
        return self.model(observations)

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
        # Need to first convert observation into Variable
        obs_variable = Variable(torch.from_numpy(observation),
            volatile=True).type(torch.FloatTensor)

        action = self.forward(obs_variable)

        return action.data.numpy(), dict()

    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()
