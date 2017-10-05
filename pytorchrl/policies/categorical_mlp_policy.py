import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from rllab.misc.overrides import overrides

from pytorchrl.core.parameterized import Parameterized
from pytorchrl.distributions.categorical import Categorical
from pytorchrl.policies.base import StochasticPolicy


class CategoricalMLPPolicy(StochasticPolicy, Parameterized):
    """

    """
    def __init__(
        self,
        observation_dim,
        num_actions,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.Tanh,
    ):
        """
        """
        super(CategoricalMLPPolicy, self).__init__()
        self.observation_dim = int(observation_dim)
        self.num_actions = num_actions

        # Create the network
        sizes = [self.observation_dim] + list(hidden_sizes)
        layers = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                layers.append(nn.Linear(size, sizes[index + 1]))
                layers.append(hidden_nonlinearity())

        last_hidden_size = sizes[len(sizes) - 1]
        layers.append(nn.Linear(last_hidden_size, self.num_actions))

        self.model = nn.Sequential(*layers)

        self.dist_cls = Categorical

    def forward(self, observations):
        """
        Parameters
        ----------
        observations (Variable):

        Returns
        -------
        prob (Variable):
        """
        logits = self.model(observations)
        prob = F.softmax(logits)
        return prob

    @overrides
    def get_action(self, observation, deterministic=False):
        """

        Parameters
        ----------
        observation (numpy.ndarray):
        deterministic (bool):

        Returns
        -------
        action (int):
        agent_info (dict):
        """
        obs_var = Variable(torch.from_numpy(observation).type(torch.FloatTensor),
            volatile=True)

        prob = self.forward(obs_var)
        if deterministic:
            _, action = prob.max(0)
        else:
            action = prob.multinomial()

        return action.data[0], dict(prob=prob.data.numpy())


    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()

    def get_policy_distribution(self, obs_var):
        prob = self.forward(obs_var)
        return self.dist_cls.from_dict(prob)

    def distribution(self, dist_info):
        prob = dist_info['prob']
        if isinstance(prob, np.ndarray):
            prob = Variable(torch.from_numpy(prob).type(torch.FloatTensor))

        return self.dist_cls.from_dict(prob)