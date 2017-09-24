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

    TRPO uses hessian free optimization.
    http://icml2010.haifa.il.ibm.com/papers/458.pdf
    """

    def __init__(
        self,
        step_size=0.01,
        **kwargs
    ):

        super(TRPO, self).__init__(**kwargs)


    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Optimizing the policy using TRPO.
        maxmize_{\theta}
            g * (\theta - \theta_old) - C/2 (\theta - \theta_old)^{T} F (\theta - \theta_old)

            where g is the gradient of the surrogate_loss
            and F is the hessian of KL-divergence

        solution is \theta - \theta_old = 1/C * F^{-1} * g
        """

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

        # Step 1: compute gradient g in Euclidean space
        self.policy.zero_grad()
        surr_loss = self.surrogate_loss(obs_var, action_var, adv_var, old_dist_var)
        surr_loss.backward()
        # Obtain the flattened gradient vector
        flat_grad = self.policy.get_grad_values()

        # Step 2: perform conjugate gradient to compute approximate
        # natural gradient
        logger.log('Computing natural gradient using conjugate gradient algorithm')
        # Clear the gradient value
        self.policy.zero_grad()
        kl_div = self.kl_divergence(obs_var, old_dist_var)
        kl_div.backward()
        flat_kl_grad = self.policy.get_grad_values()

        def Fx(v):
            return self.finite_diff_hv(self.policy, self.kl_divergence, flat_kl_grad, v)
        descent_direction = self.cg(Fx, flat_grad)

        # Step 3: Compute initial step size
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

    def kl_divergence(self, all_obs, old_dist):
        """
        Compute the kl divergence of the old and new dist.

        Parameters
        ----------
        all_obs (Variable):

        Returns
        -------
        kl_div (Variable):
        """
        new_dist = self.policy.get_policy_distribution(all_obs)
        old_dist = self.policy.distribution(old_dist)

        kl_div = old_dist.kl_div(new_dist).mean()
        return kl_div

    def finite_diff_hv(self, policy, kl_func, grad0, v, eps=1e-5, damping=1e-8):
        """
        TODO (ewei) this is finite difference method for computing the Hv,
        where H is the hessian and v is the vector. (See section 2 of
        "Fast Exact Multiplication by the Hessian")
        Should also implement the default method

        Approximately compute the Fisher-vector product of the provided policy,
        F(x)v, where x is the current policy parameter
        and v is the vector we want to form product with.

        Define g(x) to be the gradient of the KL divergence (kl_func)
        evaluated at x. Note that for small \\epsilon, Taylor expansion gives
        g(x + \\epsilon v) ≈ g(x) + \\epsilon F(x)v
        So
        F(x)v \\approx (g(x + \epsilon v) - g(x)) / \\epsilon
        Since x is always the current parameters, we cache the computation of g(x)
        and this is provided as an input, grad0

        Parameters
        ----------
        policy (nn.Module): The policy to compute Fisher-vector product
        kl_func (function): A function which computes the average KL divergence
        grad0 (torch.Tensor): The gradient of KL divergence evaluated at the current parameter x.
        v (): The vector we want to compute product with
        eps (): A small perturbation for finite difference computation.
        damping (): A small damping factor to ensure that the Fisher information matrix is positive definite.
        """

    def linearsearch():
        pass

    def cg(f_Ax, b, cg_iter, residual_tol=1e-10):
        """
        Conjugate gradient descent.
        Algorithm is from page 312 of book
        "Applied Numerical Linear Algebra" by James W. Demmel

        Approximately solve x = A^{-1}b, or Ax = b,
        Here A is F and b is g.

        A good thing about this algorithm is that, to compute x,
        we only need to have access of function f: d -> Ad, where
        function f takes an arbitary vector d and output the
        Fisher-vector product Ad, where A is the fisher information
        matrix, i.e. the hessian of KL divergence.

        This method uses krylov space solver.
        See
        "A Brief Introduction to Krylov Space Methods for Solving Linear Systems"
        http://www.sam.math.ethz.ch/~mhg/pub/biksm.pdf

        Specifically, it create a sequence of Krylov subspaces K_n for Ax = b,
        and find approximate solutions x_n in K_n

        Parameters
        ----------
        f_Ax (function): A funtion takes a vector x as input and output
            Ax where A is the Fisher Information matrix.
        b ():
        cg_iter (int): A integer, indicate the iteration for conjugate
            gradient.
        residual_tol (float): A float, indicate the terminate condition
            for the algorithm.
        """
        p = b.copy()
        r = b.copy()
        x = np.zeros_like(b)
        rdotr = r.dot(r)

        for i in range(cg_iters):
            z = f_Ax(p)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < residual_tol:
                break

        return x