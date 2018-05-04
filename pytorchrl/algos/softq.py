import copy
import time

import numpy as np
import pyprind

from rllab.algos.base import RLAlgorithm
from rllab.algos.ddpg import SimpleReplayPool
from rllab.misc import ext, special
import rllab.misc.logger as logger

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

from pytorchrl.core.kernel import AdaptiveIsotropicGaussianKernel
from pytorchrl.misc.tensor_utils import log_sum_exp
from pytorchrl.sampler import parallel_sampler

def input_bounds(inp, output):
    """
    Modifies the gradient of a given graph ('output') with respect to its
    input so that the gradient always points towards the inputs domain.
    It is assumed that the input domain is L_\infty (max-norm) unit ball (A Box).

    'input_bounds' can be used to implement the SVGD algorithm, which assumes a
    target distribution with infinite action support: 'input_bounds' allows
    actions to temporally violate the boundaries, but the modified gradient will
    eventually bring them back within boundaries.

    For example, if we have a action is 1.0711116, which is greater than 1, then
    it's gradient will be modified to -10, so it will become smaller after next update.
    If the action is -1.3577943, its gradient will be modified to 10, so next
    update will make it larger.

    Parameters
    ----------
    inp (Variable): Input tensor with a constrained domain.
    output (Variable): Output tensor, whose gradient will be modified.
    """
    violation = torch.max(torch.abs(inp) - 1, 0)
    total_violation = torch.sum(violation, dim=-1, keepdim=True)
    expanded_total_violation = total_violation * Variable(torch.ones_like(output))
    # TODO (ewei) unfinished
    pass



def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length,))
    rewards = np.zeros((path_length,))
    all_infos = list()
    t = 0
    for t in range(t, path_length):

        action, _ = policy.get_action(observation)
        next_obs, reward, terminal, info = env.step(action)

        all_infos.append(info)

        actions[t, :] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t, :] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    last_obs = observation

    concat_infos = dict()
    for key in all_infos[0].keys():
        all_vals = [np.array(info[key])[None] for info in all_infos]
        concat_infos[key] = np.concatenate(all_vals)

    path = dict(
        last_obs=last_obs,
        dones=terminals[:t+1],
        actions=actions[:t+1],
        observations=observations[:t+1],
        rewards=rewards[:t+1],
        env_infos=concat_infos
    )

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = list()
    for i in range(n_paths):
        paths.append(rollout(env, policy, path_length))

    return paths


class SoftQ(RLAlgorithm):
    """

    """
    def __init__(
        self,
        env,
        policy,
        qf,
        batch_size=64,
        n_epochs=1000,
        epoch_length=1000,
        min_pool_size=10000,
        replay_pool_size=1000000,
        discount=0.99,
        alpha=1,
        max_path_length=250,
        qf_target_n_particles=16,
        qf_weight_decay=0.,
        qf_update_method=optim.Adam,
        qf_learning_rate=1e-3,
        policy_update_method=optim.Adam,
        policy_learning_rate=1e-3,
        kernel_class=AdaptiveIsotropicGaussianKernel,
        kernel_n_particles=16,
        kernel_update_ratio=0.5,
        n_eval_episodes=10,
        soft_target_tau=1000,
        scale_reward=1.0,
        include_horizon_terminal_transitions=False,
        plot=False,
    ):
        """

        NOTE: for each of the Tensor or Variable, we make the shape of them at
            the end. And typically, we have N as the batch size.

        Parameters
        ----------
        qf_target_n_particles (int): Number of uniform samples used to estimate
            the soft target value for TD learning.
        policy_learning_rate (float): SVGD learning rate
        kernel_n_particles (int): Total number of particles per state used in
            the SVGD updates.
        alpha (int): TODO (ewei), verify this:
            this seems have no relationship with the alpha
            in soft q learning, but in svgd, see eq (21) in SVGD draw sample paper
            Learning to draw samples with amortized stein variational gradient
            descent.
        """
        self.env = env
        self.observation_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.policy = policy
        self.qf = qf
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.discount = discount
        self.max_path_length = max_path_length
        self.n_eval_episodes = n_eval_episodes
        self.soft_target_tau = soft_target_tau
        self.scale_reward = scale_reward
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.alpha = alpha
        self.plot = plot

        # 'Fixed particles' are used to compute the inner empirical expectation
        # in the SVGD update (first kernel argument); 'updated particles' are
        # used to compute the outer empirical expectation (second kernel
        # argument).
        # See Equation in Appendix C.1 for detail
        self.kernel_K = kernel_n_particles
        self.kernel_update_ratio = kernel_update_ratio

        self.kernel_K_updated = int(self.kernel_K * self.kernel_update_ratio)
        self.kernel_K_fixed = self.kernel_K - self.kernel_K_updated

        self.qf_target_K = qf_target_n_particles

        self.kernel_class = kernel_class

        # Target network
        self.target_qf = copy.deepcopy(self.qf)

        # Define optimizer
        self.qf_optimizer = qf_update_method(self.qf.parameters(),
            lr=qf_learning_rate, weight_decay=qf_weight_decay)
        self.policy_optimizer = policy_update_method(self.policy.parameters(),
            lr=policy_learning_rate)

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)


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

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
                if terminal:  # or path_length > self.max_path_length:
                    # Note that if the last time step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    observation = self.env.reset()
                    self.policy.reset()
                    path_length = 0
                    path_return = 0

                action, _ = self.policy.get_action(observation)

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

            if self.plot:
                rollout(self.env, self.policy,
                    path_length=self.max_path_length,
                    render=True,
                    speedup=2)

        self.env.terminate()


    def do_training(self, itr, batch):
        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            'observations', 'actions', 'rewards', 'next_observations',
            'terminals'
        )

        # Training in github code is parallel
        # Although in paper, we train qf first then policy,
        # We follow the convention in paper.
        qf_loss, qvals, ys = self.train_qf(obs, actions, rewards, next_obs, terminals)
        self.train_policy(obs)

        # Update target policy here
        if itr % self.soft_target_tau == 0:
            self.target_qf.set_param_values(self.qf.get_param_values())

        self.qf_loss_averages.append(qf_loss)
        # self.policy_surr_averages.append(policy_surr)
        self.q_averages.append(qvals)
        self.y_averages.append(ys)


    def train_qf(self, obs, actions, rewards, next_obs, terminals):
        """
        Train q function

        Parameters
        ----------
        obs (numpy.ndarray):
        actions (numpy.ndarray):
        rewards (numpy.ndarray):
        next_obs (numpy.ndarray):
        terminals (numpy.ndarray):

        Returns
        -------
        loss (numpy.ndarray):
        q_val (numpy.ndarray):
        ys (numpy.ndarray):
        """
        obs_var = Variable(torch.from_numpy(obs).type(torch.FloatTensor)) # N * obs_dim
        actions_var = Variable(torch.from_numpy(actions).type(torch.FloatTensor)) # N * act_dim

        # Actions are normalized, and should reside between -1 and 1. The
        # environment will clip the actions, so we'll encode that as a prior
        # directly into the Q-function.
        clipped_actions_var = torch.clamp(actions_var, -1, 1) # N * act_dim
        q_curr = self.qf(obs_var, clipped_actions_var) # N * 1
        q_curr = q_curr.squeeze() # N

        next_obs_var = Variable(torch.from_numpy(next_obs).type(
            torch.FloatTensor)) # N * obs_dim
        next_obs_var_expanded = torch.unsqueeze(next_obs_var, 1) # N * 1 * obs_dim

        # Value of the next state is approximated with uniform samples
        target_actions = Variable(torch.FloatTensor(
            1, self.qf_target_K, self.action_dim).uniform_(-1, 1)) # 1 * self.qf_target_K * act_dim

        q_next = self.target_qf(next_obs_var_expanded, target_actions) # N * self.qf_target_K * 1

        n_target_particles = q_next.size()[1]

        # Eq. (10):
        # if we replace the expectation with empricial average, then we get
        # v_next(s') = \alpha log{ 1/n * \sum_{a'} exp{1/\alpha Q_soft(s',a')} / q(a') }
        # since we are using uniform distribution for q(a'), thus, for all a',
        # there q(a') is equal which is (1 / (b - a))^(act_dim). Where b and a is the upper
        # and lower bound of action space, which is 1 and -1. Thus, we have
        # v_next(s') = \alpha log{ \sum_{a'} exp{1/\alpha Q_soft(s',a')} / (n * q(a')) }
        # v_next(s') = \alpha {log{ \sum_{a'} exp{1/\alpha Q_soft(s',a')}} - log(n) - log{q(a')}}
        # v_next(s') = \alpha {log{ \sum_{a'} exp{1/\alpha Q_soft(s',a')}} - log(n) - act_dim * log(b - a)}
        # Since b - a = 2, we have
        # v_next(s') = \alpha {log{ \sum_{a'} exp{1/\alpha Q_soft(s',a')}} - log(n) - act_dim * log(2)}
        v_next = log_sum_exp(q_next, 1).squeeze() # N

        # Importance weights add just a constant to the value, which is
        # irrelevant in terms of the actual policy.
        v_next = v_next - Variable(torch.log(torch.FloatTensor([n_target_particles])))
        v_next = v_next + self.action_dim * Variable(torch.log(torch.FloatTensor([2])))

        # Qhat_soft in Eq. (11):
        # TODO (ewei): dimension is not matching so far. And where the alpha go in Eq (10)?
        rewards_var = Variable(torch.from_numpy(rewards).type(torch.FloatTensor)) # N
        terminals_var = Variable(torch.from_numpy(terminals).type(torch.FloatTensor)) # N
        ys = rewards_var + (1.0 - terminals_var) * self.discount * v_next # N
        # Do not backpropagate into target network
        ys = ys.detach()


        # Eq (11):
        # J_Q(\theta) = E[1/2 (Q^{hat}_{soft}(s, a) - Q_{soft}(s, a))^2]
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_curr, ys) # 1

        # Backpropagation and gradient descent
        self.qf_optimizer.zero_grad()
        loss.backward()
        self.qf_optimizer.step()
        return loss.data.numpy(), q_curr.data.numpy(), ys.data.numpy()

    def train_policy(self, obs):
        """
        original equation for stein gradient descent.

        phi(z_i) = 1/n \sum_{j=1}^{n} [\nabla_{z_j} log p(z_j) k(z_j, z_i) +
                                    \nabla_{z_j} k(z_j, z_i)]

        Parameters
        ----------
        obs (numpy.ndarray):
        """
        obs_var = Variable(torch.from_numpy(obs).type(torch.FloatTensor)) # N * obs_dim

        # The particle sample used to evalute the inner empirical distribution
        actions_fixed_from_policy = self.policy(obs_var, self.kernel_K_fixed) # N * K_fix * act_dim
        # The particles samples used to evalute the outer empirical distribution
        actions_updated = self.policy(obs_var, self.kernel_K_updated) # N * K_upd * act_dim

        # The gradients should not be back-propagated into the inner
        # empirical expectation.
        # TODO (ewei) Not totally sure what is the point of requiring grad
        # for action_fixed in original code, but we do it here any way.
        # However, if we are using detach, the pytorch will complain that
        # detached Variable do not require gradient, thus, we do it here
        # a naive way.
        actions_fixed = Variable(actions_fixed_from_policy.data, requires_grad=True)
        def invert_grad(x):
            """
            Use as a trick to bring the outbound action inside the bound.
            See docstring of input_bounds function.
            """
            new_grad = x.clone()
            greater_index = actions_fixed > 1
            lower_index = actions_fixed < -1
            new_grad[greater_index] = -10
            new_grad[lower_index] = 10
            return new_grad

        actions_fixed.register_hook(invert_grad)

        # SVGD target Q-function. Expand the dimensions to make use of
        # broadcasting (see documentation for NeuralNetwork). This will
        # evaluate the Q-function for each state-action input pair.
        obs_var_expanded = obs_var.unsqueeze(1) # N x 1 x obs_dim
        q_unbounded = self.qf(obs_var_expanded, actions_fixed) # N * 1 * obs_dim


        # Target log-density in original equation (see docstring):
        # which is also Q_soft in eq. (13):
        log_p = q_unbounded # N * K_fix * 1
        # \nabla_z log p(z) = \nabla_a Q_soft(s,a)/α
        # Return of grad is a tuple with only one element
        grad_log_p = torch.autograd.grad(
            log_p,
            actions_fixed,
            grad_outputs=torch.ones(log_p.size())
        )[0] # N * K_fix * act_dim

        grad_log_p = grad_log_p.unsqueeze(2) # N * K_fix * 1 * act_dim
        grad_log_p = grad_log_p.detach()

        # Kernel function in eq. (13).
        kernel = self.kernel_class(actions_fixed, actions_updated)
        kappa = kernel.kappa # N * K_fix * K_updated
        kappa = kappa.unsqueeze(3) # N x K_fix x K_upd x 1

        kappa_grads = kernel.grad # N x K_fix x K_upd x act_dim

        # Stein Variational Gradient! Eq. (13):
        action_grads = torch.mean(
            kappa * grad_log_p # N * K_fix * K_upd * act_dim
            + self.alpha * kappa_grads,
            dim=1,
        )  # N * K_upd * act_dim

        # Propagate the gradient through the policy network. Eq. (14):
        # The original equation is Eq. (15) in paper:
        # Learning to draw samples with amortized stein variational
        # gradient descent
        self.policy_optimizer.zero_grad()
        torch.autograd.backward(
            -actions_updated,
            grad_variables=action_grads)
        self.policy_optimizer.step()


    def evaluate(self, epoch, pool):
        logger.log("Collecting samples for evaluation")

        paths = rollouts(self.env, self.policy,
                         self.max_path_length, self.n_eval_episodes)

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)

        qfun_param_norm = self.qf.get_param_values().norm()

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('StdReturn', np.std(returns))
        logger.record_tabular('MaxReturn', np.max(returns))
        logger.record_tabular('MinReturn', np.min(returns))
        logger.record_tabular('AverageDiscountedReturn', average_discounted_return)
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff', np.mean(np.abs(all_qs - all_ys)))
        logger.record_tabular('QFunParamNorm', qfun_param_norm)

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []


    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            target_qf=self.target_qf)




