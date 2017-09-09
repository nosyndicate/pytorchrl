

class GaussianMLPPolicy(Policy, Parameterized):
    """
    Stochastic policy as Gaussian distribution.
    """
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.ReLU
    ):
        """
        Create Policy Network
        """

        Super(GaussianMLPPolicy, self)__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Define network
        sizes = [int(self.observation_dim)] + list(hidden_sizes)
        submodules = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                submodules.append(nn.Linear(size, sizes[index + 1]))
                submodules.append(hidden_nonlinearity())


        self.base = nn.Sequential(*submodules)
        # Add the last layer
        submodules.append(nn.Linear(sizes[len(sizes) - 1], int(self.action_dim)))
        submodules.append(nn.Tanh())

