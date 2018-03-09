import torch
from torch.autograd import Variable
import torch.nn as nn

from rllab.misc.overrides import overrides

from pytorchrl.core.parameterized import Parameterized
from pytorchrl.policies.base import Policy


class NormalizedAdvantageFunction(Policy, Parameterized):
    def __init__(
        self,
        observation_dim,
        action_dim,
        vf_hidden_sizes=(32, 32),
        vf_nonlinearity=nn.ReLU,
        mean_hidden_sizes=(32, 32),
        mean_nonlinearity=nn.ReLU,
        pds_hidden_sizes=(32, 32),
        pds_nonlinearity=nn.ReLU,
    ):

        """
        Create the Normalizad Advantage Function

        Note, although NAF is conceptually Q-function, we can actually
        call get_action on it, thus, we set this as the subclass of
        Policy.

        Parameters
        ----------
        observation_dim (int): dimensionality of observation space
        action_dim (int): dimensionality of action space.
        vf_hidden_size (tuple): sizes of hidden layers of value fuction V,
             every two successive integer define a hidden layer.
        vf_nonlinearity (nn.Module): Module for activation function.
        mean_hidden_size (tuple): sizes of hidden layers of mean of advantage
            fuction A.
        mean_nonlinearity (nn.Module): activation function for the
            mean of the quadractic function
        pds_hidden_size (tuple): sizes of hidden layers for generating elements
            in positive-definite square matrix of advantage fuction A.
        covariance_nonlinearity (nn.Module): activation function for
            positive-definite square matrix of advantage fuction A.

        """
        super(NormalizedAdvantageFunction, self).__init__()

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        # Define network
        self.v = self.define_single_network(vf_hidden_sizes, vf_nonlinearity, 1)
        # TODO (ewei), the nn.Tanh seems not used, need to verified with paper
        self.mu = self.define_single_network(mean_hidden_sizes, mean_nonlinearity,
            self.action_dim, output_nonlinearity=nn.Tanh)
        self.l_output = int(self.action_dim * (self.action_dim + 1) / 2)

        # Mask is used to form lower triangular matrix
        # Since mask is used in forward function, it has to
        # be a Variable instead of Tensor
        self.formation_mask = Variable(torch.ones(self.action_dim,
            self.action_dim).type(torch.ByteTensor).tril())
        self.strict_lower_tri_mask = Variable(torch.ones(self.action_dim,
            self.action_dim).tril(-1))
        self.diag_mask = Variable(torch.eye(self.action_dim))

        self.l = self.define_single_network(
            pds_hidden_sizes, pds_nonlinearity, self.l_output)

        # Initialization

    def define_single_network(self, sizes, nonlinearity, output_size,
        output_nonlinearity=None):
        """
        Define a single input - single output network (thus, no shared
        part).

        Parameters
        ----------
        sizes (list):
        nonlinearity (nn.Module):
        output_size (int): size of the final output
        output_nonlinearity (nn.Module): activation function apply to
            the last layer's output

        Returns
        -------
        model (nn.Module):
        """
        sizes = [self.observation_dim] + list(sizes)
        layers = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                layers.append(nn.Linear(size, sizes[index + 1]))
                layers.append(nonlinearity())

        # Add last layer
        layers.append(nn.Linear(sizes[len(sizes) - 1], output_size))
        return nn.Sequential(*layers)

    def forward(self, obs_var, actions_var):
        """
        Return Q(s, a) for given state and action

        Parameters
        ----------
        obs_var (Variable): state wrapped in Variable
        actions_var (Variable): action wrapped in Variable

        Returns
        -------
        q_var (Variable): Q value of the state and action,
            wrapped in Variable
        """
        # V(s)
        v_var = self.forward_v(obs_var)
        # \mu(s)
        mu_var = self.forward_mu(obs_var)
        # L(s)
        l_var = self.forward_l(obs_var)
        size = len(obs_var)

        # Scatter the output of l network into lower triangular matrice
        zero_matrices = Variable(torch.zeros(size, self.action_dim, self.action_dim))
        l_matrices_var = zero_matrices.masked_scatter_(self.formation_mask, l_var)

        # Exponential the diagonal elements
        l_matrices_var = l_matrices_var * self.strict_lower_tri_mask \
            + torch.exp(l_matrices_var) * self.diag_mask

        # P(s) = L(s) * L(s)^T
        p_var = torch.bmm(l_matrices_var, l_matrices_var.transpose(2, 1))

        # u - \mu(s)
        u_minus_mu = (actions_var - mu_var).unsqueeze(2)

        # A(s,a) = -1/2 (a - mu(s))^T * P(s) * (a - mu(s))
        a_var = -0.5 * torch.bmm(
            torch.bmm(u_minus_mu.transpose(2, 1), p_var), u_minus_mu)[:,:,0]

        # Q(s,a) = A(s,a) + V(s)
        q_var = v_var + a_var

        return q_var

    def forward_v(self, obs_variable):
        """
        Return the state value V(s) for the observation.

        Parameters
        ----------
        observations (Variable): state wrapped in Variable

        Returns
        -------
        v (Variable): state value given the observation.
        """
        return self.v(obs_variable)

    def forward_mu(self, obs_variable):
        """
        Return the mu value (mean action) for the observation.

        Parameters
        ----------
        observations (Variable): state wrapped in Variable

        Returns
        -------
        mean_action (Variable): mean action given the state
        """
        return self.mu(obs_variable)

    def forward_l(self, obs_variable):
        """
        Returnt the elements for lower-triangular matrix, which is
        used to compute the positive-definite square matrix in quadratic
        term.

        Parameters
        ----------
        observations (Variable): state wrapped in Variable

        Returns
        -------
        elements (Variable): elements of the lower-triangular matrix
        """
        return self.l(obs_variable)

    def get_stateval(self, observation):
        """
        Return V(s）for given state and action. This is used for
        determine the V(s') on target network. Thus, we do not
        need to call backward on the model. So we set the Variable's
        volatile to True.

        Parameters
        ----------
        observation (numpy.ndarray): state

        Returns
        -------
        state_val (numpy.ndarray): V value of the state
        """
        obs_variable = Variable(torch.from_numpy(observation).type(
            torch.FloatTensor), volatile=True)

        state_value_variable = self.forward_v(obs_variable)
        return state_value_variable.data.numpy()

    def get_action(self, observation):
        """
        Get observation and return the action for the observation.

        Parameters
        ----------
        observation (numpy.ndarray): current observation.

        Return
        ------
        action (numpy.ndarray): action for the observation.
        other_info (dict): Additional info for the action
        """
        obs_variable = Variable(torch.from_numpy(observation).type(
            torch.FloatTensor), volatile=True)

        action = self.forward_mu(obs_variable)
        return action.data.numpy(), dict()

    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()
