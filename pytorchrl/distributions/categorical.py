import numpy as np

import torch

from pytorchrl.distributions.base import Distribution

class Categorical(Distribution):
    def __init__(self, prob, tiny=1e-8):
        """
        Parameters
        ----------
        prob (Variable):
        """
        self.prob = prob
        self.tiny = tiny

    @classmethod
    def from_dict(cls, prob):
        return cls(prob=prob)

    def entropy(self):
        """
        In case of P(x_i) = 0 for some i, the corresponding summand
        0 * log(0) is taken to be 0, which is consistent witht the limit:
        \lim_{p->0} p * log(p) = 0.
        See https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Thus, we just set the log(x_i) to some arbitary value in computation
        when x_i = 0.
        """
        prob_data = self.prob.data.numpy()
        alter_prob_data = prob_data.copy()
        alter_prob_data[prob_data == 0.0] = 1
        return -np.sum(prob_data * np.log(alter_prob_data), axis=1)

    def log_likelihood(self, a):
        """
        Compute the log likelihood of a.

        Parameters
        ----------
        a (Variable):

        Returns
        -------
        logli (Variable):
        """
        # TODO (ewei) need to verify if following statement is correct
        # We do not need to take derivate with respect to a, thus we convert
        # it into Tensor
        action_indices = a.data.type(torch.LongTensor)
        length = len(action_indices)
        sample_indices = torch.arange(0, length).type(torch.LongTensor)
        likelihood = self.prob[sample_indices, action_indices]
        logli = (likelihood + self.tiny).log()
        return logli

    def kl_div(self, other):
        """
        Parameters
        ----------
        other (Categorical):

        Returns
        -------
        kl_div (Variable):
        """
        kl_div = (self.prob *
            ((self.prob + self.tiny).log() - (other.prob + other.tiny).log())
            ).sum(-1)
        # print(kl_div)
        return kl_div



