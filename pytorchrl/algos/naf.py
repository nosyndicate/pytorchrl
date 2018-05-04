import copy

import numpy as np
import pyprind

from rllab.algos.base import RLAlgorithm
from rllab.algos.ddpg import SimpleReplayPool
from rllab.misc import special
from rllab.misc import ext
import rllab.misc.logger as logger

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

from pytorchrl.sampler import parallel_sampler
from pytorchrl.sampler.utils import rollout


class NAF(RLAlgorithm):
    """
    Normalized Advantage Function.
    """
    def __init__(
        self,
        env,
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
        eval_samples=10000,
        soft_target=True,
        soft_target_tau=0.001,
        n_updates_per_sample=5,
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
        self.es = es
        self.batch_size = batch_size
        self.n_epoch = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.discount = discount
        self.max_path_length = max_path_length
        self.qf_weight_decay = qf_weight_decay
        # The update method and learning are using below
        self.eval_samples = eval_samples
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions

        self.plot = plot
        self.pause_for_plot = pause_for_plot

        # Target network
        self.target_qf = copy.deepcopy(self.qf)

        # Define optimizer
        self.qf_optimizer = qf_update_method(self.qf.parameters(),
            lr=qf_learning_rate)

        self.qf_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.scale_reward = scale_reward

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.qf)

    def train(self):
        pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim
        )

        self.start_worker()

        itr = 0
        path_length = 0
        path_return = 0
        terminal = False
        # Seed the environment
        self.env.seed(np.random.randint(2**32-1))
        observation = self.env.reset()

        sample_policy = copy.deepcopy(self.qf)

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
                    sample_policy.reset()
                    self.es_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0

                action = self.es.get_action(itr, observation, policy=sample_policy)

                next_observation, reward, terminal, _ = self.env.step(action)
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
                    sample_policy.set_param_values(self.qf.get_param_values())

                itr += 1

            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                self.evaluate(epoch, pool)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            if self.plot:
                rollout(self.env, self.qf, animated=True,
                    max_path_length=self.max_path_length,
                    speedup=2)
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.env.terminate()

    def do_training(self, itr, batch):
        # Update Q Function
        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        next_qvals = self.target_qf.get_stateval(next_obs)

        rewards = rewards.reshape(-1, 1)
        terminals_mask = (1.0 - terminals).reshape(-1, 1)
        ys = rewards + terminals_mask * self.discount * next_qvals

        qf_loss = self.train_qf(ys, obs, actions)

        self.target_qf.set_param_values(
            self.target_qf.get_param_values() * (1 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)

    def train_qf(self, expected_qval, obs_val, actions_val):
        """
        Given the mini-batch, fit the Q-value

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

        q_vals = self.qf(obs, actions)

        # Define loss function
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_vals, expected_q)

        # Backpropagation and gradient descent
        self.qf_optimizer.zero_grad()
        loss.backward()
        self.qf_optimizer.step()

        return loss.data.numpy()

    def evaluate(self, epoch, pool):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.qf.get_param_values(),
            max_samples=self.eval_samples,
            max_path_length=self.max_path_length,
        )
        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        # all_qs = np.concatenate(self.q_averages)
        # all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        # policy_reg_param_norm = np.linalg.norm(
        #     self.policy.get_param_values(regularizable=True)
        # )
        # qfun_reg_param_norm = np.linalg.norm(
        #     self.qf.get_param_values(regularizable=True)
        # )

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('AverageReturn',
                              np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageQLoss', average_q_loss)
        # logger.record_tabular('AverageQ', np.mean(all_qs))
        # logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        # logger.record_tabular('AverageY', np.mean(all_ys))
        # logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        # logger.record_tabular('AverageAbsQYDiff',
        #                       np.mean(np.abs(all_qs - all_ys)))
        logger.record_tabular('AverageAction', average_action)

        # logger.record_tabular('PolicyRegParamNorm',
        #                       policy_reg_param_norm)
        # logger.record_tabular('QFunRegParamNorm',
        #                       qfun_reg_param_norm)

        # TODO (ewei), may need to add log_diagnostics method
        # in policy class
        # self.env.log_diagnostics(paths)
        # self.policy.log_diagnostics(paths)

        self.qf_loss_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            target_qf=self.target_qf,
            es=self.es)
