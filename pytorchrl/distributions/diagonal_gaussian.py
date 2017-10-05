import numpy as np

import torch

from pytorchrl.distributions.base import Distribution
from pytorchrl.misc.tensor_utils import constant


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

        Returns
        -------
        logli (Variable)
        """
        # First cast into float tensor
        a = a.type(torch.FloatTensor)

        # Convert into a sample of standard normal
        zs = (a - self.means) / (self.log_stds.exp())

        # TODO (ewei), I feel this equation is not correct.
        # Mainly the first line
        # TODO (ewei), still need to understand what is meaning of having
        # -1 for axis in sum method, (same for numpy)
        logli = - self.log_stds.sum(-1) - \
            constant(0.5) * zs.pow(2).sum(-1) - \
            constant(0.5) * constant(float(self.dim)) * constant(float(np.log(2 * np.pi)))

        return logli

    def kl_div(self, other):
        """
        Given the distribution parameters of two diagonal multivariate Gaussians,
        compute their KL divergence (vectorized)

        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Kullback.E2.80.93Leibler_divergence_for_multivariate_normal_distributions

        In general, for two n-dimensional distributions, we have

        D_KL(N1||N2) =
        1/2 ( tr(Σ_2^{-1}Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) - n + ln(det(Σ_2) / det(Σ_1)) )

        Here, Σ_1 and Σ_2 are diagonal. Hence this equation can be simplified.
        In terms of the parameters of this method,

        determinant of diagonal matrix is product of diagonal, thus
        - ln(det(Σ_2) / det(Σ_1)) = sum(2 * (log_stds_2 - log_stds_1), axis=-1)

        inverse of diagonal matrix is the diagonal matrix of elements at diagonal inverted, thus
        - (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) = sum((means_1 - means_2)^2 / vars_2, axis=-1)

        trace is sum of the diagonal elements
        - tr(Σ_2^{-1}Σ_1) = sum(vars_1 / vars_2, axis=-1)

        Where

        - vars_1 = exp(2 * log_stds_1)

        - vars_2 = exp(2 * log_stds_2)

        Combined together, we have

        D_KL(N1||N2)
        = 1/2 ( tr(Σ_2^{-1}Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) - n + ln(det(Σ_2) / det(Σ_1)) )
        = sum(1/2 * ((vars_1 - vars_2) / vars_2 + (means_1 - means_2)^2 / vars_2 + 2 * (log_stds_2 - log_stds_1)), axis=-1)
        = sum( ((means_1 - means_2)^2 + vars_1 - vars_2) / (2 * vars_2) + (log_stds_2 - log_stds_1)), axis=-1)

        Parameters
        ----------
        other (DiagonalGaussian):

        Returns
        -------
        kl_div (Variable):
        """
        # Constant should wrap in Variable to multiply with another Variable
        # TODO (ewei) kl seems have problem
        variance = (constant(2.0) * self.log_stds).exp()
        other_variance = (constant(2.0) * other.log_stds).exp()
        numerator = (self.means - other.means).pow(2) + \
            variance - other_variance
        denominator = constant(2.0) * other_variance + constant(1e-8)
        # TODO (ewei), -1 for sum has a big impact, need to figure out why
        kl_div = (numerator / denominator + other.log_stds - self.log_stds).sum(-1)

        return kl_div
