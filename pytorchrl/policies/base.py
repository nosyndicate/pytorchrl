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

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def recurrent(self):
        """
        Indicate whether the policy is recurrent.
        """
        return False

    def terminate(self):
        """
        Clean up operation
        """
        pass


class StochasticPolicy(Policy):

    @property
    def distribution(self):
        """
        Get the parameters of the policy distribution as Variables.
        """
        raise NotImplementedError
