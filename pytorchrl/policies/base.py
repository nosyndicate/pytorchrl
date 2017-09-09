import torch.nn as nn


class Policy(nn.Module):
    """
    Base Policy class
    """
    def __init__(self):
        """
        Initialize the Module
        """
        super(Policy, self).__init__()

    def get_action(self, observation):
        """
        Get observation and return the action for the observation.

        Parameters
        ----------
        observation (numpy.ndarray): current observation.
        """
        raise NotImplementedError

    def reset(self):
        pass


class StochasticPolicy(Policy):

    def distribution(self):
        """
        Get the parameters of the policy distribution as Variables.
        """
        raise NotImplementedError
