"""
Implementation of SVG(0) in using pytorch
"""

import argparse
import gym
import numpy as np


import torch
import torch.nn as nn
import torhc.nn.functional as F
import torch.autograd as  autograd
from torhc.autograd import Variable



class SVG(RLAlgorithm):
    """
    SVG(0)
    """

    def __init__(
        self,
        env,
        policy,
        qf,
        batch_size=32,
        n_epoch=200,
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
        scale_reward=1.0,
        include_horizon_terminal_transitions=False,
        plot=False,
        pause_for_plot=False,
    ):


        self.env = env
        self.observation_dim = np.prod(env.obervation_space.shape)
        self.action_dim = np.prod(env.action_space.shape)
        self.policy = policy
        self.qf = qf
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.discount = discount
        self.max_path_length = max_path_length
        self.qf_weight_decay = qf_weight_decay
        self.eval_samples = eval_samples

        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions

        self.plot = plot
        self.pause_for_plot = pause_for_plot

        # Target network
        self.target_qf = copy.deepcopy(self.qf)

        # Define optimizer
        self.qf_optimizer = qf_update_method(self.qf.parameters(),
            lr=qf_learning_rate, weight_decay=self.qf_weight_decay)
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


    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)

    def train(self):
        pool = SARSAReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim
        )

        self.start_worker()

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
                    path_length = 0
                    path_return = 0

                noise = self.sample_noise()
                action = policy.get_action_with_noise(noise)
                # clip the action
                action = np.clip(action + ou_state, self.env.action_space.low,
                    self.action_space.high)

                next_observation, reward, terminal, _ = self.env.step(action)
                path_length += 1
                path_return += reward

                if not terminal and path_length >= self.max_path_length:
                    terminal = True
                    # only include the terminal transition in this case if the
                    # flag was set.

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
                    sample_policy.set_param_values(self.policy.get_param_values())

                itr += 1

            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                self.evaluate(epoch, pool)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            if self.plot:
                rollout(self.env, self.policy, animated=True,
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

        next_actions, _ = self.target_policy.get_action(next_obs)
        next_qvals = self.target_qf.get_qval(next_obs, next_actions)

        rewards = rewards.reshape(-1, 1)
        terminals_mask = (1.0 - terminals).reshape(-1, 1)
        ys = rewards + terminals_mask * self.discount * next_qvals

        qf_loss = self.train_qf(ys, obs, actions)
        policy_surr = self.train_policy(obs)

        self.target_policy.set_param_values(
            running_average_tensor_list(
                self.target_policy.get_param_values(),
                self.policy.get_param_values(),
                self.soft_target_tau))

        self.target_qf.set_param_values(
            running_average_tensor_list(
                self.target_qf.get_param_values(),
                self.qf.get_param_values(),
                self.soft_target_tau))

        self.qf_loss_averages.append(qf_loss)
        self.policy_surr_averages.append(policy_surr)

    def train_qf(self, expected_qval, obs_val, actions_val):
        """
        """
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

        return average_q.data.numpy()

    def evaluate(self, epoch, pool):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
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
        average_policy_surr = np.mean(self.policy_surr_averages)
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
        logger.record_tabular('AveragePolicySurr', average_policy_surr)
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
        self.policy_surr_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            target_qf=self.target_qf,
            target_policy=self.target_policy,
            es=self.es)
