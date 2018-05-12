import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from rllab.misc.overrides import overrides

from pytorchrl.core.parameterized import Parameterized
from pytorchrl.misc.tensor_utils import log_sum_exp


class AIRLDiscriminator(nn.Module, Parameterized):
    """
    This is the discriminator function from
    Adversarial Inverse Reinforcement Learning
    """
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.ReLU,
    ):
        """
        Create the Discriminator network

        Parameters
        ----------
        observation_dim (int): dimensionality of observation space
        action_dim (int): dimensionality of action space.
        hidden_sizes (tuple): sizes of hidden layer, every two successive
            integer define a hidden layer.
        hidden_nonlinearity (nn.Module): Module for activation function.
        """

        super(AIRLDiscriminator, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Define network
        layers = []
        sizes = [int(self.observation_dim + self.action_dim)] + list(hidden_sizes)
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                layers.append(nn.Linear(size, sizes[index + 1]))
                layers.append(hidden_nonlinearity())
        # Add the last layer
        layers.append(nn.Linear(sizes[len(sizes) - 1], 1))
        self.model = nn.Sequential(*layers)

        # TODO (ewei), should we explicite initialize here?

    def get_reward(self, obs_variable, actions_variable):
        """
        Return the reward function f(s,a) for a given
        state action pair. Note this will return a Variable rather
        than the data in the numpy format. To get reward in numpy array,
        use prediction.


        Parameters
        ----------
        obs_variable (Variable): state wrapped in Variable
        actions_variable (Variable): action wrapped in Variable

        Returns
        -------
        reward (Variable): energy function the state action pair.
        """
        state_action = torch.cat((obs_variable, actions_variable), 1)
        # The separate log Z(s) cannot be learned as it is impossible to
        # separate it from the energy function
        reward = self.model(state_action)

        # TODO (ewei), why original code use -energy, cost function?
        return reward


    def forward(self, obs_variable, action_variable, log_q_tau_var, target):
        r"""
        Return the loss function of the discriminator to be optimized.

        As in discriminator, we only want to discriminate the expert from
        learner, thus this is a binary classification problem.

        Unlike Discriminator used for GAIL, the discriminator in this class
        take a specific form, where

                        exp{f(s, a)}
        D(s,a) = -------------------------
                  exp{f(s, a)} + \pi(a|s)

        to make it consistent with GAN_GCL (Guided Cost Learning GAN formulation),
        we denote state action pair <s,a> as \tau, thus
        D(s,a) = D(\tau)

        We also denote exp{f(s,a)} = p(s,a) or p(\tau),
        and \pi(a|s) = q(s,a) or q(\tau)


        Parameters
        ----------
        obs_variable (Variable): state wrapped in Variable, of size N * obs_dim,
            where N is twice the batch size (we stack positive and negative example
            together).
        actions_variable (Variable): action wrapped in Variable, of size N * act_dim,
            where N is twice the batch size
        log_q_tau_var (Variable): log(q(s,a)）value wrapped in Variable, of size N * 1,
            where N is twice the batch size
        target (Variable): 1 or 0, mark the real and fake of the, same size as log_q_tau_var
            samples. Fake sample (0) first, real sample (1) last

        Returns
        -------
        loss (Variable):
        """
        # Rename it
        log_q_tau = log_q_tau_var

        # Get the log p(\tau)
        log_p_tau = self.get_reward(obs_variable, action_variable)

        # Concatenate the log p(\tau) and log q(\tau) to compute sum
        # After concatenation, log_pq should have size N * 2
        log_pq = torch.cat((log_p_tau, log_q_tau_var), 1)

        # Since log_pq is of size N * 2, we should sum each row, thus
        # the answer is of size N * 1
        log_pq = log_sum_exp(log_pq, 1, keepdim=True)

        # Binary Cross Entropy Loss
        # loss_n= - [y_n * log(p(x_n)) + (1 − y_n) * log(1 − p(x_n))]
        # where x_n is the input, and y_n is the output

        total_loss = (target * (log_p_tau - log_pq) + (1 - target) * (log_q_tau - log_pq))
        loss = -torch.mean(total_loss)

        return loss


    def prediction(self, observations, actions):
        """
        Make the prediction of the class label

        Parameters
        ----------
        observation (numpy.ndarray): state
        action (numpy.ndarray): action

        Returns
        -------
        reward (numpy.ndarray): A scalar, the estimated reward
        """
        obs_var = Variable(torch.from_numpy(observations),
            volatile=True).type(torch.FloatTensor)

        # obs_var sets volatile to True, thus, we do not set
        # it here
        acts_var = Variable(torch.from_numpy(actions)).type(
            torch.FloatTensor)

        reward = self.get_reward(obs_var, acts_var)

        return reward.data.numpy()


    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()
