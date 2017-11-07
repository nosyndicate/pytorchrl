import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

from rllab.misc.overrides import overrides

from pytorchrl.policies.base import Policy
from pytorchrl.core.parameterized import Parameterized

import sys

class SVGDPolicy(Policy, Parameterized):
    """
    SVGDPolicy's functions like SVGDMLPQFunction, but it feeds an additional
    random vector to the network. The shape of this vector is

    ... x K x (number of output units)

    where ... denotes all leading dimensions of the inputs (all but the last
    axis), and K is the number of outputs samples produced per data point.

    Example:

    input 1 shape: ... x D1
    input 2 shape: ... x D2
    output shape: ... x K x (output shape)

    If K == 1, the corresponding axis is dropped. Note that the leading
    dimensions of the inputs should be consistent (supports broadcasting).
    """


    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes,
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
    ):

        super(SVGDPolicy, self).__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        # Following two layers for the first layer concatenation
        self.observation_layer = nn.Linear(self.observation_dim, hidden_sizes[0], bias=False)
        self.noise_layer = nn.Linear(self.action_dim, hidden_sizes[0], bias=False)
        # Initialize the concatenation layers
        nn.init.xavier_uniform(self.observation_layer.weight)
        nn.init.xavier_uniform(self.noise_layer.weight)

        self.concate_layer_bias = nn.Parameter(torch.zeros(hidden_sizes[0]))

        # First add in the nonlinearity layer for the first concat layer
        self.rest_layers = [hidden_nonlinearity()]

        # Add the other layers and initialize them
        sizes = list(hidden_sizes)
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                layer = nn.Linear(size, sizes[index + 1])
                nn.init.xavier_uniform(layer.weight)
                nn.init.constant(layer.bias, 0)
                self.rest_layers.append(layer)
                self.rest_layers.append(hidden_nonlinearity())

        # Add the last layer and initialize it
        layer = nn.Linear(sizes[len(sizes) - 1], self.action_dim)
        nn.init.xavier_uniform(layer.weight)
        nn.init.constant(layer.bias, 0)
        self.rest_layers.append(layer)

        if output_nonlinearity is not None:
            self.rest_layers.append(output_nonlinearity())
        self.rest_layers = nn.Sequential(*self.rest_layers)


    def forward(self, obs_var, k):
        """
        See docstring of this class.

        Parameters
        ----------
        obs_var (Variable): observation wrapped in Variable
        k (int): number of action sample generate for each observation.
        """

        n_dims = obs_var.dim()
        sample_shape = np.ones(n_dims + 1, dtype=np.int32)
        sample_shape[n_dims-1:] = (k, self.action_dim)
        sample_shape = torch.Size(sample_shape.tolist())

        expanded_inputs = []
        obs_var_expanded = obs_var.unsqueeze(n_dims-1) # N * 1 * obs_dim
        expanded_inputs.append(obs_var_expanded)
        xi = torch.randn(sample_shape) # N * K * act_dim
        expanded_inputs.append(Variable(xi))

        self.xi = xi # For debugging

        # Apply the concate layer
        concate_layers = [self.observation_layer, self.noise_layer]
        result = self.concate_layer_bias
        for var, layer in zip(expanded_inputs, concate_layers):
            result = result + layer(var)

        # Apply the rest layers
        result = self.rest_layers(result) # N * K * act_dim

        return result

    def get_action(self, observation):
        """
        Parameters
        ----------
        observation (numpy.ndarray):
        """
        obs_var = Variable(torch.from_numpy(observation).type(
            torch.FloatTensor), volatile=True)

        action = self.forward(obs_var, k=1) # 1 * act_dim
        # Drop the addtional dimension
        action = action[0] # 1

        return action.data.numpy(), dict()

    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()


def main():
    policy = SVGDPolicy(
        observation_dim=3,
        action_dim=2,
        hidden_sizes=(100, 100),
        output_nonlinearity=None,
    )


if __name__ == '__main__':
    main()