import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from rllab.misc.overrides import overrides

from pytorchrl.reward_functions.discriminator import Discriminator
from pytorchrl.irl.imitation_learning import SingleTimestepIRL
from pytorchrl.misc.utils import TrainingIterator

class GAIL(SingleTimestepIRL):
    """
    Generative Adverserail imitation learning
    See https://arxiv.org/pdf/1606.03476.pdf

    This version consumes single timesteps.
    """

    def __init__(
        self,
        env_spec,
        expert_trajs=None,
        discriminator_class=Discriminator,
        discriminator_args={},
        optimizer_class=optim.Adam,
        discriminator_lr=1e-3,
        favor_zero_expert_reward=True,
    ):
        """
        Parameters
        ----------
        env_spec (): Environment
        expert_trajs (): The demonstration from experts, the trajectories sampled from
            experts policy, positive samples used to train discriminator
        discriminator_class (): Discriminator class
        discriminator_args (dict): dict of arguments for discriminator
        optimizer_class (): Optimizer class
        discrmininator_lr (float): learning rate for discriminator
        favor_zero_expert_reward (boolean):
            This is a really important flag for making some environment work, however,
            I am still not understand why (#TODO, understand why this work and how to apply
            to AIRL).
            if it's true, then
            0 for expert-like states, goes to -inf for non-expert-like states
            compatible with envs with traj cutoffs for good (expert-like) behavior
            e.g. mountain car, which gets cut off when the car reaches the destination
            otherwise
            0 for non-expert-like states, goes to +inf for expert-like states
            compatible with envs with traj cutoffs for bad (non-expert-like) behavior
            e.g. walking simulations that get cut off when the robot falls over
        """
        super(GAIL, self).__init__()
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim

        self.favor_zero_expert_reward = favor_zero_expert_reward
        # Get expert's trajectories
        self.expert_trajs = self.extract_paths(expert_trajs)

        # Build energy model
        self.discriminator = discriminator_class(
            self.obs_dim, self.action_dim, **discriminator_args)

        self.optimizer = optimizer_class(self.discriminator.parameters(),
            lr=discriminator_lr)


    @overrides
    def fit(self, trajs, batch_size=32, max_itrs=100, **kwargs):
        """
        See doc of the same function in airl.py
        """
        obs, acts = self.extract_paths(trajs)
        expert_obs, expert_acts = self.expert_trajs

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            obs_batch, act_batch = self.sample_batch(obs, acts, batch_size=batch_size)
            expert_obs_batch, expert_act_batch = self.sample_batch(expert_obs, expert_acts, batch_size=batch_size)

            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0

            # Stack the sample from learner and expert
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)


            obs_var = Variable(torch.from_numpy(obs_batch)).type(torch.FloatTensor)
            act_var = Variable(torch.from_numpy(act_batch)).type(torch.FloatTensor)
            label_var = Variable(torch.from_numpy(labels)).type(torch.FloatTensor)

            loss = self.discriminator(obs_var, act_var, label_var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            it.record('loss', loss.data.numpy())

            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        return mean_loss

    @overrides
    def eval(self, paths, **kwargs):
        """
        See doc of the same function in airl.py
        """
        obs, acts = self.extract_paths(paths)
        scores = self.discriminator.prediction(obs, acts)

        if self.favor_zero_expert_reward:
            # reward = log D(s, a)
            # Add small constant to prevent 0 reward
            scores = np.log(scores[:, 0] + 1e-8)
        else:
            # reward = -log(1 - D(s, a))
            scores = -np.log(1 - scores[:, 0] + 1e-8)

        return self.unpack(scores, paths)

    @overrides
    def get_params(self):
        """
        Get the value of the model parameters in one flat Tensor

        Returns
        -------
        params (torch.Tensor): A torch Tensor contains the parameter of
            the module.
        """
        return self.discriminator.get_param_values()

    @overrides
    def set_params(self, params):
        """
        Set the value of model parameter using parameters.

        Parameters
        ----------
        params (torch.Tensor): A tensors, this new_parameters should
            have the same format as the return value of get_param_values
            method.
        """
        self.discriminator.set_param_values(params)