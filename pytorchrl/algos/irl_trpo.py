import numpy as np
import torch
from torch.autograd import Variable

from rllab.misc.overrides import overrides
from rllab.misc import ext
import rllab.misc.logger as logger

from pytorchrl.algos.irl_batch_polopt import IRLBatchPolopt
# import helpful function from trpo implementation
from pytorchrl.algos.trpo import (
    kl_divergence,
    finite_diff_hvp,
    pearlmutter_hvp,
    cg,
    line_search
)



def surrogate_loss(policy, all_obs, all_actions, all_adv, old_dist, policy_ent):
    r"""
    Compute the loss of policy evaluated at current parameter
    given the observation, action, and advantage value. Since we
    are dealing with maximum entropy RL in IRL, we add policy entropy
    regularization term.

    In the TRPO step for IRL, we are maximizing
    E_{\pi}[\nabla_{\theta} \log \pi_{\theta}(a|s) Q(s,a)]
    + \lambda \nabla_{\theta} H(\pi_{\theta})

    where H(\pi_{\theta}) = E_{\pi}[- \log \pi_{\theta}(a|s)]

    The last term according to Lemma A.1 in GAIL paper.
    E_{\pi}[- \log \pi_{\theta}(a|s)]
    = E_{\pi}[\nabla_{\theta} \log \pi_{\theta}(a|s) Q_{log}(s,a)]
    where Q_{log}(s,a) = E_{\pi}[- \log \pi_{\theta}(a|s)]
    and - \log \pi_{\theta}(a|s) can be thought as additional reward term,
    which is the entropy of policy \pi(a|s), H(\pi{\theta}(a|s))

    We can combine the two expectation terms, we get
    E_{\pi}[\nabla_{\theta} \log \pi_{\theta}(a|s) Q(s,a) + \lambda H(\pi_{\theta}(a|s)]

    When we use GAE (Generalized Advantage Estimation) to estimate Q(s,a)
    and add entropy of policy to the Q(s,a) for optimization.

    One critical difference between AIRL and GAIL is that in GAIL,
    we optimize the policy and consider entropy part of the optimization goal.
    \pi is depend on \theta. In AIRL, this term is treat as filling value,
    thus we do not backpropagate through thisterm. (See A.3 in AIRL paper, where
    \pi do not have subscript \theta)

    However, since GAIL do not use entropy term (0 for \lambda), it's ok
    to unify them.


    Parameters
    ----------
    policy (nn.Module):
    all_obs (Variable):
    all_actions (Variable):
    all_adv (Variable):
    old_dist (dict): The dict of means and log_stds Variables of
        collected samples
    policy_ent (float): coefficient for policy entropy regularization term

    Returns
    -------
    surr_loss (Variable): The surrogate loss function wrapped in
        Variable
    """

    new_dist = policy.get_policy_distribution(all_obs)
    old_dist = policy.distribution(old_dist)

    ratio = new_dist.likelihood_ratio(old_dist, all_actions)

    # Add the entropy in optimization goal
    if policy_ent > 0:
        entropy = Variable(torch.from_numpy(new_dist.entropy()).type(torch.FloatTensor))
        adv = all_adv + policy_ent * entropy
    else:
        adv = all_adv

    surr_loss = -(ratio * adv).mean()

    return surr_loss


class IRLTRPO(IRLBatchPolopt):
    """
    Inverse Reinforcement Learning using
    TRPO as forward (RL) algorithm
    """

    def __init__(
        self,
        step_size=0.01,
        entropy_weight=1.0,
        damping=1e-8,
        use_line_search=True,
        use_finite_diff_hvp=False,
        symmetric_finite_diff=True,
        finite_diff_hvp_epsilon=1e-5,
        **kwargs
    ):

        """
        Inverse Reinforcement Learning with TRPO.

        Parameters
        ----------
        step_size (float): This is the constraint on the kl-divergence,
            that is D_KL(old||new) <= step_size
        entropy_weight (float):
        damping (float): A small damping factor to ensure that the Fisher
            information matrix is positive definite.
        use_line_search (bool): This controls whether we want to perform
            a line search after we find out the descent direction. It not,
            TRPO become Truncated Natural Policy Gradient algorithm.
        use_finite_diff_hvp (bool): Controls whether to use finite difference
            method or Pearlmutter's method to acquire hessian-vector product.
        symmetric_finite_diff (bool): Control whether to use symmetric
            finite difference method to compute hessian-vector product.
        """
        super(IRLTRPO, self).__init__(**kwargs)
        self.step_size = step_size
        self.pol_ent_wt = entropy_weight
        self.damping = damping
        self.use_line_search = use_line_search
        self.use_finite_diff_hvp = use_finite_diff_hvp
        self.symmetric_finite_diff = symmetric_finite_diff
        self.finite_diff_hvp_epsilon = finite_diff_hvp_epsilon

    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Optimizing the policy using TRPO.
        maxmize_{\theta}
        g * (\theta - \theta_old) - C/2 (\theta - \theta_old)^{T} F (\theta - \theta_old)

        where g is the gradient of the surrogate_loss
        and F is the hessian of KL-divergence

        solution is \theta - \theta_old = 1/C * F^{-1} * g

        Parameters
        ----------
        itr ():
        samples_data ():

        Returns
        -------
        info ():
        """

        # TODO (ewei), rllab code use agent_info to store the dict return
        # by get_action method, this is really confusing. May need to
        # think to rename it.

        # samples_data contains the observation, action, and other information
        # for all the samples we collected, the size of the samples_data is
        # specified by batch_size argument in BatchPolopt class
        # old_dist will contains the means and log_stds for all the steps in
        # samples_data
        all_obs, all_actions, all_adv, old_dist = ext.extract(
            samples_data, 'observations', 'actions', 'advantages', 'agent_infos')

        # Convert all the data into Variable, we do not need the gradient of
        # them.
        obs_var = Variable(torch.from_numpy(all_obs).type(torch.FloatTensor))
        action_var = Variable(torch.from_numpy(all_actions).type(torch.FloatTensor))
        adv_var = Variable(torch.from_numpy(all_adv).type(torch.FloatTensor))

        old_dist_var = {
            k: Variable(torch.from_numpy(v).type(torch.FloatTensor))
            for k, v in old_dist.items()
        }

        # Step 1: compute gradient g in Euclidean space
        self.policy.zero_grad()
        surr_loss = surrogate_loss(self.policy, obs_var, action_var, adv_var, old_dist_var, self.pol_ent_wt)
        surr_loss.backward()
        # Obtain the flattened gradient vector
        flat_grad = self.policy.get_grad_values()

        # Step 2: perform conjugate gradient to compute approximate
        # natural gradient
        logger.log('Computing natural gradient using conjugate gradient')

        flat_kl_grad = None
        if self.use_finite_diff_hvp and not self.symmetric_finite_diff:
            # We first compute the gradient of KL term evaluate at current
            # parameter, this is used multiple times in non symmetric
            # finite difference method
            self.policy.zero_grad()
            kl_div = kl_divergence(self.policy, obs_var, old_dist_var)
            kl_div.backward()
            flat_kl_grad = self.policy.get_grad_values()

        def Fx(vector):
            if self.use_finite_diff_hvp:
                hvp = finite_diff_hvp(kl_divergence, obs_var,
                    old_dist_var, self.policy, flat_kl_grad, vector,
                    eps=self.finite_diff_hvp_epsilon,
                    symmetric=self.symmetric_finite_diff)
            else:
                hvp = pearlmutter_hvp(kl_divergence, obs_var,
                    old_dist_var, self.policy, vector)

            return hvp + self.damping * vector

        descent_step = cg(Fx, flat_grad)

        # Step 3: Compute initial step size
        # We'd like D_KL(old||new) <= step_size
        # The 2nd order approximation gives 1/2 * d^T * H * d <= step_size,
        # where d is the descent step
        # Hence given the initial direction d_0 we can rescale it
        # so that this constraint is tight
        # Let this scaling factor be \alpha, i.e. d = \alpha * d_0
        # Solving 1/2 * \alpha^2 * d_0^T * H * d_0 = step_size
        # we get \alpha = \sqrt(2 * step_size / d_0^T * H * d_0)
        # add 1e-8 to avoid divided by zero
        scale = torch.sqrt(torch.Tensor([
            2.0 * self.step_size *
            (1.0 / (descent_step.dot(Fx(descent_step)) + 1e-8))
        ]))

        if np.isnan(scale.numpy()[0]):
            logger.log('Scale is nan!!!!!!')
            scale = torch.Tensor([1])

        tight_descent_step = descent_step * scale

        current_parameters = self.policy.get_param_values()

        # Step 4: perform line search
        if self.use_line_search:
            logger.log('Performing line search')
            expected_improvement = flat_grad.dot(tight_descent_step)

            def barrier_func(x):
                """
                barrier function to perform search on

                Parameters
                ----------
                x (torch.Tensor): The parameter of the policy

                Returns
                -------
                f(x) (float): The loss function evaluated given x
                """
                self.policy.set_param_values(x)
                # This full inference mode, we do not need gradient
                obs_var.volatile = True
                surr_loss = surrogate_loss(self.policy, obs_var, action_var, adv_var, old_dist_var, self.pol_ent_wt)
                kl_div = kl_divergence(self.policy, obs_var, old_dist_var)
                constraint_part = max(kl_div.data[0] - self.step_size, 0.0)
                result = surr_loss.data[0] + 1e100 * constraint_part
                return result

            new_parameters = line_search(barrier_func,
                x0=current_parameters, dx=tight_descent_step,
                y0=surr_loss.data[0], expected_improvement=expected_improvement)

        else:
            new_parameters = current_parameters - descent_step

        self.policy.set_param_values(new_parameters)

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