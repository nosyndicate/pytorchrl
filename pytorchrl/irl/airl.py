import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from rllab.misc.overrides import overrides

from pytorchrl.reward_functions.airl_discriminator import AIRLDiscriminator
from pytorchrl.irl.fusion_manager import RamFusionDistr
from pytorchrl.irl.imitation_learning import SingleTimestepIRL
from pytorchrl.misc.utils import TrainingIterator

class AIRL(SingleTimestepIRL):
    """
    Adversarial Inverse Reinforcement Learning
    Similar to GAN_GCL except operates on single timesteps.
    """
    def __init__(
        self,
        env_spec,
        expert_trajs=None,
        discriminator_class=AIRLDiscriminator,
        discriminator_args={},
        optimizer_class=optim.Adam,
        discriminator_lr=1e-3,
        l2_reg=0,
        discount=1.0,
        fusion=False,
    ):
        """
        Parameters
        ----------
        env_spec (): Environment
        expert_trajs (): The demonstration from experts, the trajectories sampled from
            experts policy, positive samples used to train discriminator
        discriminator_class (): Discriminator class
        optimizer_class (): Optimizer class
        discrmininator (float): learning rate for discriminator
        discriminator_args (dict): dict of arguments for discriminator
        l2_reg (float): L2 penalty for discriminator parameters
        discount (float): discount rate for reward
        fusion (boolean): whether use trajectories from old iterations to train.
        """
        super(AIRL, self).__init__()
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim

        # Get expert's trajectories
        self.expert_trajs = expert_trajs
        # expert_trajs_extracted will be a single matrix with N * dim
        # where N is the total sample number, and dim is the dimension of
        # action or state
        self.expert_trajs_extracted = self.extract_paths(expert_trajs)

        # Build energy model
        self.discriminator = discriminator_class(
            self.obs_dim, self.action_dim, **discriminator_args)

        self.optimizer = optimizer_class(self.discriminator.parameters(),
            lr=discriminator_lr, weight_decay=l2_reg)

        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=0.5)
        else:
            self.fusion = None



    def fit(self, trajs, policy=None, batch_size=32, max_itrs=100, logger=None, **kwargs):
        """
        Train the AIRL Discriminator to distinguish expert from learner.

        Parameters
        ----------
        trajs (list): trajectories sample from current policy
        batch_size (integer): How many positive sample or negative samples
        max_iters (integer): max iteration number for training discriminator
        kwargs (dict):

        Returns
        -------
        mean_loss (numpy.ndarray): A scalar indicate the average discriminator
            loss (Binary Cross Entropy Loss) of total max_iters iterations
        """
        if self.fusion is not None:
            # Get old trajectories
            old_trajs = self.fusion.sample_paths(n=len(trajs))
            # Add new trajectories for future use
            self.fusion.add_paths(trajs)
            # Mix old and new trajectories
            trajs = trajs + old_trajs

        # Eval the prob of the trajectories under current policy
        # The result will be fill in part of the discriminator
        self.eval_trajectory_probs(trajs, policy, insert=True)
        self.eval_trajectory_probs(self.expert_trajs, policy, insert=True)

        obs, acts, trajs_probs = self.extract_paths(trajs, keys=('observations', 'actions', 'a_logprobs'))
        expert_obs, expert_acts, expert_probs = self.extract_paths(self.expert_trajs, keys=('observations', 'actions', 'a_logprobs'))

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):

            # lprobs_batch is the prob of obs and act under current policy
            obs_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs, acts, trajs_probs, batch_size=batch_size)

            # expert_lprobs_batch is the experts' obs and act under current policy
            expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs, expert_acts, expert_probs, batch_size=batch_size)

            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)

            # obs_batch is of size N * obs_dim, act_batch is of size N * act_dim
            # where N is the batch size. However, lprobs_batch if of size N.
            # Thus, we need to use expand_dims to reshape it to N * 1
            lprobs_batch = np.expand_dims(
                np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0),
                axis=1).astype(np.float32)

            obs_var = Variable(torch.from_numpy(obs_batch)).type(torch.FloatTensor)
            act_var = Variable(torch.from_numpy(act_batch)).type(torch.FloatTensor)
            label_var = Variable(torch.from_numpy(labels)).type(torch.FloatTensor)
            lprobs_var = Variable(torch.from_numpy(lprobs_batch)).type(torch.FloatTensor)

            loss = self.discriminator(obs_var, act_var, lprobs_var, label_var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            it.record('loss', loss.data.numpy())

            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        if logger:
            reward = self.discriminator.prediction(obs, acts)
            logger.record_tabular('IRLAverageReward', np.mean(reward))
            logger.record_tabular('IRLAverageLogQtau', np.mean(trajs_probs))
            logger.record_tabular('IRLMedianLogQtau', np.median(trajs_probs))

            reward = self.discriminator.prediction(expert_obs, expert_acts)
            logger.record_tabular('IRLAverageExpertReward', np.mean(reward))
            logger.record_tabular('IRLAverageExpertLogQtau', np.mean(expert_probs))
            logger.record_tabular('IRLMedianExpertLogQtau', np.median(expert_probs))

        return mean_loss


    def eval(self, paths, **kwargs):
        """
        Return the estimate reward of the paths.

        Parameters
        ----------
        paths (list): See doc of eval_trajectory_probs method

        Returns
        -------
        rewards (list): estimated reward arrange according to unpack method.

        """
        obs, acts = self.extract_paths(paths)

        rewards = self.discriminator.prediction(obs, acts)
        rewards = rewards[:,0]

        return self.unpack(rewards, paths)

    def eval_trajectory_probs(self, paths, policy, insert=False):
        """
        Evaluate probability of paths under current policy

        Parameters
        ----------
        paths (list): Each element is a dict. Each dict represent a whole
            trajectory, contains observations, actions, rewards, env_infos,
            agent_infos. observations and actions is of size T * dim, where T
            is the length of the trajectory, and dim is the dimension of observation
            or action. rewards is a vector of length T. agent_infos contains other
            information about the policy. For example, when we have a Gaussian policy,
            it may contain the means and log_stds of the Gaussian policy at each step.
        policy (pytorchrl.policies.base.Policy): policy object used to evaluate
            the probabilities of trajectories.
        insert (boolean): Whether to insert the action probabilities back into
            the paths
        """
        for path in paths:
            actions, agent_infos = policy.get_actions(path['observations'])
            path['agent_infos'] = agent_infos
        return self.compute_path_probs(paths, insert=insert)


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


