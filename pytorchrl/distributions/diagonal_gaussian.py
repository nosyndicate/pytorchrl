import numpy as np

import torch
from torch.autograd import Variable

from pytorchrl.distributions.base import Distribution


class DiagonalGaussian(Distribution):
    """
    Instead of a distribution, rather a collection of distribution.
    """
    def __init__(self, means, log_stds):
        """
        Parameters
        ----------
        means (Variable):
        log_stds (Variable):
        """
        self.means = means
        self.log_stds = log_stds

        # dim is the dimension of action space
        self.dim = self.means.size()[-1]

    @classmethod
    def from_dict(cls, means, log_stds):
        """
        Parameters
        ----------
        means (Variable):
        log_std (Variable):
        """
        return cls(means=means, log_stds=log_stds)

    def entropy(self):
        """
        Entropy of gaussian distribution is given by
        1/2 * log(2 * \pi * e * sigma^2)
        = log(sqrt(2 * \pi * e) * sigma))
        = log(sigma) + log(sqrt(2 * \pi * e))
        """
        return np.sum(self.log_stds.data.numpy() + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    def log_likelihood(self, a):
        """
        Compute log likelihood of a.

        Parameters
        ----------
        a (Variable):
        """
        # First cast into float tensor
        a = a.type(torch.FloatTensor)

        # Convert into a sample of standard normal
        zs = (a - self.means) / (self.log_stds.exp())

        # pytorch require multiplication take either two variables
        # or two tensors. And it is recommended to wrap a constant
        # in a variable which is kind of silly to me.
        # https://discuss.pytorch.org/t/adding-a-scalar/218
        constant = float(np.log(2 * np.pi))
        constant_part = Variable(torch.Tensor([constant])).type(torch.FloatTensor)
        dimension_part = Variable(torch.Tensor([self.dim])).type(torch.FloatTensor)
        half = Variable(torch.Tensor([0.5])).type(torch.FloatTensor)

        # TODO (ewei), we feel this equation is not correct.
        # Mainly the first line
        return - self.log_stds.sum() - \
            half * zs.pow(2).sum() - \
            half * dimension_part * constant_part

    def kl(self, other):
        pass

