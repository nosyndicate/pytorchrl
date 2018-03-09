import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from rllab.misc.overrides import overrides

from pytorchrl.core.parameterized import Parameterized

class Discriminator(nn.Module, Parameterized):
    """
    This is the discriminator function in the
    Generative Adversarial Imitation Learning setting.
    Discriminator take a state-action pair and output the
    probability that is a sample from learner.
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
        super(Discriminator, self).__init__()

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

        # TODO (ewei), should we explicit initialize here?


    def get_logits(self, obs_variable, actions_variable):
        """
        Return the logit for a given state action pair

        Parameters
        ----------
        obs_variable (Variable): state wrapped in Variable
        actions_variable (Variable): action wrapped in Variable

        Returns
        -------
        logits (Variable): Logits of the state action pair. If we
            apply exp to the logit, we recover the
            probability of state-action are being real (or fake).
        """
        state_action = torch.cat((obs_variable, actions_variable), 1)
        logits = self.model(state_action)

        return logits


    def forward(self, obs_variable, actions_variable, target):
        """
        Compute the cross entropy loss using the logit, this is more numerical
        stable than first apply sigmoid function and then use BCELoss.

        As in discriminator, we only want to discriminate the expert from
        learner, thus this is a binary classification problem.

        Parameters
        ----------
        obs_variable (Variable): state wrapped in Variable
        actions_variable (Variable): action wrapped in Variable
        target (Variable): 1 or 0, mark the real and fake of the
            samples

        Returns
        -------
        loss (Variable):
        """
        logits = self.get_logits(obs_variable, actions_variable)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, target)

        return loss


    def prediction(self, observation, action):
        """
        Make the prediction of the class label

        Parameters
        ----------
        observation (numpy.ndarray): state
        action (numpy.ndarray): action

        Returns
        -------
        prob (numpy.ndarray):
        """
        obs_variable = Variable(torch.from_numpy(observation),
            volatile=True).type(torch.FloatTensor)

        # obs_variable sets volatile to True, thus, we do not set
        # it here
        action_variable = Variable(torch.from_numpy(action)).type(
            torch.FloatTensor)

        logits = self.get_logits(obs_variable, action_variable)
        probs = F.sigmoid(logits)

        return probs.data.numpy()

    @overrides
    def get_internal_params(self):
        return self.parameters()

    @overrides
    def get_internal_named_params(self):
        return self.named_parameters()