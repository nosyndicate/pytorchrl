import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

from rllab.misc.overrides import overrides

from pytorchrl.core.parameterized import Parameterized

class SVGDMLPQFunction(nn.Module, Parameterized):
    """
    The Q function in Soft Q-Learning for Stein Variational Gradient Descent.

    Multilayer Perceptron that support broadcasting.

    Creates a multi-layer perceptron with given hidden sizes. A nonlinearity
    is applied after every hidden layer.

    Supports input tensors of rank 2 and rank 3. All inputs should have the same
    tensor rank. It is assumed that the vectors along the last axis are the
    data points, and an mlp is applied independently to each leading dimension.
    If multiple inputs are provided, then the corresponding rank-1 vectors
    are concatenated along the last axis. The leading dimensions of the network
    output are equal to the 'outer product' of the inputs' shapes.

    Example:

    input 1 shape: N x K x D1
    input 2 shape: N x 1 x D2

    output shape: N x K x (number of output units)
    """


    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes,
        hidden_nonlinearity=nn.ReLU
    ):

        super(SVGDMLPQFunction, self).__init__()

        # Take the above example,
        # if input for observation is of shape N * K * obs_dim
        # and input for action is of shape N * 1 * act_dim
        # after apply to their corresponding weights maxtrix W,
        # which is of shape obs_dim * first_layer_hidden_size,
        # and act_dim * first_layer_hidden_size,
        # then the output for each input is N * K * first_layer_hidden_size
        # and N * 1 * first_layer_hidden_size
        # Then we add them up, due to the broadcast, we have
        # N * K * first_layer_hidden_size
        # then we apply each layer to this out
        # so the output will be:
        # N * K * second_layer_hidden_size
        # ....
        # N * K * last_layer_hidden_size

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        # Following two layers for the first layer concatenation
        self.observation_layer = nn.Linear(self.observation_dim, hidden_sizes[0], bias=False)
        self.action_layer = nn.Linear(self.action_dim, hidden_sizes[0], bias=False)

        # Initialize it
        nn.init.xavier_uniform(self.observation_layer.weight)
        nn.init.xavier_uniform(self.action_layer.weight)

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
        layer = nn.Linear(sizes[len(sizes) - 1], 1)
        nn.init.xavier_uniform(layer.weight)
        nn.init.constant(layer.bias, 0)
        self.rest_layers.append(layer)
        self.rest_layers = nn.Sequential(*self.rest_layers)


    def forward(self, obs_var, act_var):
        """

        Parameters
        ----------
        obs_var (Variable): observation wrapped in Variable
        act_var (Variable): action wrapped in Variable
        """
        inputs = [obs_var, act_var]
        concate_layers = [self.observation_layer, self.action_layer]
        # Apply the concate layer
        result = self.concate_layer_bias

        for var, layer in zip(inputs, concate_layers):
            result = result + layer(var)

        # Apply the rest layers
        result = self.rest_layers(result)

        return result


    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()



def main():
    qf = SVGDMLPQFunction(3, 2, (3, 3))


if __name__ == '__main__':
    main()








