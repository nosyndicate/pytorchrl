import torch
from torch.autograd import Variable

from rllab.misc.overrides import overrides
from rllab.misc import ext
import rllab.misc.logger as logger

from pytorchrl.algos.batch_polopt import BatchPolopt

import sys

class TRPO(BatchPolopt):
    """
    Trust region policy optimization.
    """

    def __init__(
        self,
        step_size=0.01,
        **kwargs
    ):

        super(TRPO, self).__init__(**kwargs)


    @overrides
    def optimize_policy(self, itr, samples_data):

        # TODO (ewei), rllab code use agent_info to store the dict return
        # by get_action method, this is really confusing. May need to
        # think to rename it.
        all_obs, all_actions, all_adv, old_dist = ext.extract(
            samples_data, 'observations', 'actions', 'advantages', 'agent_infos')

        # Convert all the data into Variable, we do not need the gradient of
        # them.
        obs_var = Variable(torch.from_numpy(all_obs)).type(torch.FloatTensor)
        action_var = Variable(torch.from_numpy(all_actions)).type(torch.FloatTensor)
        adv_var = Variable(torch.from_numpy(all_adv)).type(torch.FloatTensor)
        old_dist_var = {
            k: Variable(torch.from_numpy(v)).type(torch.FloatTensor)
            for k, v in old_dist.items()
        }
        # First compute gradient in Euclidean space
        self.policy.zero_grad()
        surr_loss = self.surrogate_loss(obs_var, action_var, adv_var, old_dist_var)
        surr_loss.backward()

        # Obtain the flattened gradient vector
        # flat_grad = self.policy.get_flat_grad()


        # # Second, perform conjugate gradient to compute approximate
        # # natural gradient
        # logger.log('Computing approximate natural gradient using conjugate gradient algorithm')
        # # Clear the gradient value
        # self.policy.zero_grad()
        # kl_div = self.kl_divergence()
        # kl_div.backward()
        # flat_kl_grad = self.policy.get_flat_grad()
        # descent_direction = self.cg()

        # new_param = self.linesearch()

        # self.policy.set_param_values(new_params)




        # logger.record_tabular('LossBefore', loss_before)
        # logger.record_tabular('LossAfter', loss_after)
        # logger.record_tabular('MeanKLBefore', mean_kl_before)
        # logger.record_tabular('MeanKL', mean_kl)
        # logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    def surrogate_loss(self, all_obs, all_actions, all_adv, old_dist):
        """
        Compute the loss given the observation, action, and advantage value

        Parameters
        ----------
        all_obs (Variable):
        all_actions (Variable):
        all_adv (Variable):

        Returns
        -------
        surr_loss (Variable):
        """
        new_dist = self.policy.get_policy_distribution(all_obs)
        old_dist = self.policy.distribution(old_dist)

        ratio = new_dist.likelihood_ratio(old_dist, all_actions)
        surr_loss = -(ratio * all_adv).mean()

        return surr_loss






    def fisher_vector_product(policy, f_kl, grad0, v, eps=1e-5, damping=1e-8):
        """
        Approximately compute the Fisher-vector product of the provided policy,
        F(x)v, where x is the current policy parameter and v is the vector we
        want to form product with.

        Define g(x) to be the gradient of the KL divergecne (f_kl) evaluated at
        x.

        g(x + \epsilon v)
        """

    def linearsearch():
        pass

    def cg(f_Ax, b, cg_iter, residual_tol=1e-10):
        """
        Conjugate gradient descent.
        """
        pass