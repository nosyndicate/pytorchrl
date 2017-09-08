import copy

import numpy as np
import pyprind

from rllab.algos.base import RLAlgorithm
from rllab.algos.ddpg import SimpleReplayPool
from rllab.misc import ext
import rllab.misc.logger as logger

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

class Parameterized(object):
    def get_internal_params(self):
        raise NotImplementedError

    def get_param_values(self):
        """
        Get the values of model parameters.

        Returns
        -------
        param_tensors (list) : A list contains tensors which is the
            clone of the parameters of the model
        """
        param_tensors = [parameter.data for parameter in self.get_internal_params()]

        return param_tensors

    def set_param_values(self, new_param_tensors):
        """
        Set the value of model parameter using new_param

        Parameters
        ----------
        param_values (list) : A list of tensors, this parameter should
            have the same format as the return value of get_param_values
            method.
        """
        # First check if the dimension match
        param_tensors = [parameter.data for parameter in self.get_internal_params()]
        assert len(param_tensors) == len(new_param_tensors)
        for param, new_param in zip(param_tensors, new_param_tensors):
            assert param.shape == new_param.shape

        for param, new_param in zip(param_tensors, new_param_tensors):
            param.copy_(new_param)


class ContinuousMLPQFunction(nn.Module, Parameterized):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.ReLU
    ):
        """
        Create the Q-network.

        Parameters
        ----------
        observation_dim (int): dimensionality of observation space
        action_dim (int): dimensionality of action space.
        hidden_sizes (tuple): sizes of hidden layer, every two successive
            integer define a hidden layer.
        hidden_nonlinearity (nn.Module): Module for activation function.
        """
        super(ContinuousMLPQFunction, self).__init__()

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

        # NOTE : We didn't use He's initialzation for previous layer
        # This is different from the DDPG paper.
        for param in self.model.parameters():
            nn.init.uniform(param, -3e-3, 3e-3)

    def forward(self, obs_variable, actions_variable):
        """
        Return Q(s, a) for given state and action

        Parameters
        ----------
        obs_variable (Variable): state wrapped in Variable
        actions_variable (Variable): action wrapped in Variable

        Returns
        -------
        q_val_variable (Variable): Q value of the state and action,
            wrapped in Variable
        """
        state_action = torch.cat((obs_variable, actions_variable), 1)
        q_val_variable = self.model(state_action)

        return q_val_variable

    def get_qval(self, observation, action):
        """
        Return Q(s, a) for given state and action. This is
        used for determine the Q(s', a') on target network,
        thus, we do not need to call backward on the model.
        So we set Variables to violatile True.

        Parameters
        ----------
        observation (numpy.ndarray): state
        action (numpy.ndarray): action

        Returns
        -------
        q_val (numpy.ndarray): Q value of the state and action
        """
        obs_variable = Variable(torch.from_numpy(observation),
            volatile=True).type(torch.FloatTensor)

        # obs_variable sets volatile to True, thus, we do not set
        # it here
        action_variable = Variable(torch.from_numpy(action)).type(
            torch.FloatTensor)

        q_value_variable = self.forward(obs_variable, action_variable)
        return q_value_variable.data.numpy()

    def get_internal_params(self):
        return self.parameters()


class DeterministicMLPPolicy(nn.Module, Parameterized):
    """
    Deterministic Policy
    """
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.ReLU
    ):
        """
        Create Policy Network.
        """
        super(DeterministicMLPPolicy, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Define network
        sizes = [int(self.observation_dim)] + list(hidden_sizes)
        submodules = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                submodules.append(nn.Linear(size, sizes[index + 1]))
                submodules.append(hidden_nonlinearity())

        # Add the last layer
        submodules.append(nn.Linear(sizes[len(sizes) - 1], int(self.action_dim)))
        submodules.append(nn.Tanh())

        self.model = nn.Sequential(*submodules)

        # NOTE : We didn't use He's initialzation for previous layer
        for param in self.model.parameters():
            nn.init.uniform(param, -3e-3, 3e-3)

    def forward(self, observations):
        """
        Return action for given state

        Parameters
        ----------
        observations (Variable): state wrapped in Variable

        Returns
        -------
        action (Variable): action wrapped in Variable
        """
        return self.model(observations)

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
        # Need to first convert observation into Variable
        obs_variable = Variable(torch.from_numpy(observation),
            volatile=True).type(torch.FloatTensor)

        action = self.forward(obs_variable)

        return action.data.numpy(), dict()

    def get_internal_params(self):
        return self.parameters()


class DDPG(RLAlgorithm):
    def __init__(
        self,
        env,
        policy,
        qf,
        es,
        batch_size=32,
        n_epochs=200,
        epoch_length=1000,
        min_pool_size=10000,
        replay_pool_size=1000000,
        discount=0.99,
        max_path_length=250,
        qf_weight_decay=0.,
        qf_update_method=optim.Adam,
        qf_learning_rate=1e-3,
        policy_weight_decay=0,
        policy_update_method=optim.Adam,
        policy_learning_rate=1e-4,
        eval_samples=10000,
        soft_target=True,
        soft_target_tau=0.001,
        n_updates_per_sample=1,
        scale_reward=1.0,
        include_horizon_terminal_transitions=False,
        plot=False,
        pause_for_plot=False,
    ):
        """
        """
        self.env = env
        self.observation_dim = np.prod(env.observation_space.shape)
        self.action_dim = np.prod(env.action_space.shape)
        self.qf = qf
        self.policy = policy
        self.es = es
        self.replay_pool_size = replay_pool_size
        self.discount = discount
        self.n_epoch = n_epochs
        self.epoch_length = epoch_length
        self.max_path_length = max_path_length
        self.min_pool_size = min_pool_size
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.batch_size = batch_size
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions

        self.plot = plot
        self.pause_for_plot = pause_for_plot

        # Target network
        self.target_qf = copy.deepcopy(self.qf)
        self.target_policy = copy.deepcopy(self.policy)

        # Define optimizer
        self.qf_optimizer = qf_update_method(self.qf.parameters(),
            lr=qf_learning_rate, weight_decay=qf_weight_decay)
        self.policy_optimizer = policy_update_method(self.policy.parameters(),
            lr=policy_learning_rate)

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.scale_reward = scale_reward

    def train(self):
        pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim
        )

        itr = 0
        path_length = 0
        path_return = 0
        terminal = False
        observation = self.env.reset()

        for epoch in range(self.n_epoch):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
                # Execute policy
                if terminal:  # or path_length > self.max_path_length:
                    # Note that if the last time step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    observation = self.env.reset()
                    self.es.reset()
                    self.es_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0

                action = self.es.get_action(itr, observation, policy=self.policy)

                next_observation, reward, terminal, _ = self.env.step(action)

                if self.plot:
                    self.env.render()

                path_length += 1
                path_return += reward

                if not terminal and path_length >= self.max_path_length:
                    terminal = True
                    # only include the terminal transition in this case if the flag was set
                    if self.include_horizon_terminal_transitions:
                        pool.add_sample(observation, action, reward * self.scale_reward, terminal)
                else:
                    pool.add_sample(observation, action, reward * self.scale_reward, terminal)

                observation = next_observation

                if pool.size >= self.min_pool_size:
                    for update_itr in range(self.n_updates_per_sample):
                        # Train policy
                        batch = pool.random_batch(self.batch_size)
                        self.do_training(itr, batch)

                itr += 1

            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                self.evaluate(epoch, pool)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
        self.env.terminate()

    def do_training(self, itr, batch):
        # Update Q Function
        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        next_actions, _ = self.target_policy.get_action(next_obs)
        next_qvals = self.target_qf.get_qval(next_obs, next_actions)

        rewards = rewards.reshape(-1, 1)
        terminals_mask = (1.0 - terminals).reshape(-1, 1)
        ys = rewards + terminals_mask * self.discount * next_qvals

        self.train_qf(ys, obs, actions)
        self.train_policy(obs)

        self.target_policy.set_param_values(
            self._running_average_tensor_list(self.target_policy.get_param_values(),
                self.policy.get_param_values(), self.soft_target_tau))

        self.target_qf.set_param_values(
            self._running_average_tensor_list(self.target_qf.get_param_values(),
                self.qf.get_param_values(), self.soft_target_tau))

    def train_qf(self, expected_qval, obs_val, actions_val):
        """
        Given the mini-batch, fit the Q-value with L2 norm (defined
        in optimizer).

        Parameters
        ----------
        expected_qval (numpy.ndarray): expected q values in numpy array
            form.
        obs_val (numpy.ndarray): states draw from mini-batch, should have
            the same amount of rows as expected_qval.
        actions_val (numpy.ndarray): actions draw from mini-batch, should
            have the same amount of rows as expected_qval.
        """
        # Create Variable for input and output, we do not need gradient
        # of loss with respect to these variables
        obs = Variable(torch.from_numpy(obs_val)).type(
            torch.FloatTensor)
        actions = Variable(torch.from_numpy(actions_val)).type(
            torch.FloatTensor)
        expected_q = Variable(torch.from_numpy(expected_qval)).type(
            torch.FloatTensor)

        # Define loss function
        loss_fn = nn.MSELoss()
        loss = loss_fn(self.qf(obs, actions), expected_q)

        # Backpropagation and gradient descent
        self.qf_optimizer.zero_grad()
        loss.backward()
        self.qf_optimizer.step()

    def train_policy(self, obs_val):
        """
        Given the mini-batch, do gradient ascent on policy
        """
        obs = Variable(torch.from_numpy(obs_val)).type(torch.FloatTensor)

        # Do gradient descent, so need to add minus in front
        average_q = -self.qf(obs, self.policy(obs)).mean()

        self.policy_optimizer.zero_grad()
        average_q.backward()
        self.policy_optimizer.step()

    def _running_average_tensor_list(self, first_list, second_list, rate):
        """
        Return the result of
        first * (1 - rate) + second * rate

        Parameter
        ---------
        first_list (list) : A list of Tensors
        second_list (list) : A list of Tensors, should have the same format
            as first.
        rate (float): a learning rate, in [0, 1]

        Returns
        -------
        results (list): A list of Tensors with computed results.
        """
        results = []
        assert len(first_list) == len(second_list)
        for first_t, second_t in zip(first_list, second_list):
            assert first_t.shape == second_t.shape
            result_tensor = first_t * (1 - rate) + second_t * rate
            results.append(result_tensor)

        return results

    def evaluate(self, epoch, pool):
        pass

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            target_qf=self.target_qf,
            target_policy=self.target_policy,
            es=self.es)
