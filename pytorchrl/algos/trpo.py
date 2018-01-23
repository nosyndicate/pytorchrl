import numpy as np
import torch
from torch.autograd import Variable

from rllab.misc.overrides import overrides
from rllab.misc import ext
import rllab.misc.logger as logger

from pytorchrl.algos.batch_polopt import BatchPolopt


def surrogate_loss(policy, all_obs, all_actions, all_adv, old_dist):
    """
    Compute the loss of policy evaluated at current parameter
    given the observation, action, and advantage value

    Parameters
    ----------
    policy (nn.Module):
    all_obs (Variable):
    all_actions (Variable):
    all_adv (Variable):
    old_dist (dict): The dict of means and log_stds Variables of
        collected samples

    Returns
    -------
    surr_loss (Variable): The surrogate loss function wrapped in
        Variable
    """
    new_dist = policy.get_policy_distribution(all_obs)
    old_dist = policy.distribution(old_dist)

    ratio = new_dist.likelihood_ratio(old_dist, all_actions)
    surr_loss = -(ratio * all_adv).mean()

    return surr_loss


def kl_divergence(policy, all_obs, old_dist):
    """
    Compute the kl divergence of the old and new dist.

    Parameters
    ----------
    policy (nn.Module):
    all_obs (Variable): The observation sample wrapped in Variable
    old_dist (dict): The dict of means and log_stds Variables of
        collected samples

    Returns
    -------
    kl_div (Variable): The KL-divergence between the old distribution
        and new distribution.
    """
    new_dist = policy.get_policy_distribution(all_obs)
    old_dist = policy.distribution(old_dist)

    kl_div = old_dist.kl_div(new_dist).mean()
    return kl_div


def finite_diff_hvp(kl_func, all_obs, old_dist, policy, grad_x, v,
    eps=1e-5, symmetric=True):
    """
    This is finite difference method for computing the Hessian vector product (Hv),
    where H is the hessian and v is the vector.
    (See section 2 of "Fast Exact Multiplication by the Hessian")

    Approximately compute the Fisher-vector product of the provided policy,
    F(x)v, where x is the current policy parameter
    and v is the vector we want to form product with.

    Define g(x) to be the gradient of the KL divergence (kl_func)
    evaluated at x. Note that for small \epsilon, Taylor expansion at point x
    (derivate are computed at point x) gives
      g(x + \epsilon v)
    ≈ g(x) + \epsilon v * g'(x)
    = g(x) + \epsilon v * F(x) （g(x) is gradient, thus g'(x) should be F(x))
    = g(x) + \epsilon F(x)v
    So
    F(x)v ≈ (g(x + \epsilon v) - g(x)) / \epsilon
    Since x is always the current parameters, we cache the computation of g(x)
    and this is provided as an input, grad_x

    According to this post comment by by Barak Pearlmutter
    https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
    F(x)v ≈ (g(x + \epsilon v) - g(x - \epsilon v)) / (2 * \epsilon)
    is a better approximation, as this expression will be closer to the
    true value for larger \epsilon. Thus, we treat this symmetric evaluation
    as default setting. In this case, we do not need grad_x, thus, it can be
    set to None

    Parameters
    ----------
    kl_func (function): A function which computes the average KL divergence
    all_obs (Varialbe): observation data to compute kl_divergence
    old_dist (dict): distribution data to compute kl_divergence
    policy (nn.Module): The policy to compute Fisher-vector product
    grad_x (torch.Tensor): The gradient of KL divergence evaluated at
        the current parameters point x.
    v (torch.Tensor): The vector we want to compute product with
    eps (float): A small perturbation for finite difference computation.
    symmetric (bool): A flag to indicate if we want to use symmetric
        finite differece estimation
    Returns
    -------
    hvp (torch.Tensor): The hessian vector product
    """
    flat_params = policy.get_param_values()
    # Compute g(x + \epsilon v)
    policy.set_param_values(flat_params + eps * v)
    policy.zero_grad()
    kl_div = kl_func(policy, all_obs, old_dist)
    kl_div.backward()
    # gradient of kl_divergence evaluated at (x + \epsilon v)
    grad_plus = policy.get_grad_values()

    # Compute the finite difference
    if symmetric:
        # Compute g(x - \epsilon v)
        policy.set_param_values(flat_params - eps * v)
        policy.zero_grad()
        kl_div = kl_func(policy, all_obs, old_dist)
        kl_div.backward()
        # gradient of kl_divergence evaluated at (x - \epsilon v)
        grad_minus = policy.get_grad_values()
        hvp = (grad_plus - grad_minus) / (2 * eps)
    else:
        hvp = (grad_plus - grad_x) / (eps)

    # Restore the policy parameters
    policy.set_param_values(flat_params)
    return hvp


def pearlmutter_hvp(kl_func, all_obs, old_dist, policy, v):
    """
    TODO (ewei) add docstring here.

    Parameters
    ----------
    see docstring of finite_diff_hvp function.

    Returns
    -------
    see docstring of finite_diff_hvp function.
    """
    policy.zero_grad()
    kl_div = kl_func(policy, all_obs, old_dist)
    param_grads = torch.autograd.grad(kl_div, policy.ordered_params(),
        create_graph=True)
    flat_grad = torch.cat([grad.view(-1) for grad in param_grads])
    gradient_vector_product = torch.sum(flat_grad * Variable(v))
    hessian_vector_product = torch.autograd.grad(gradient_vector_product,
        policy.ordered_params())
    flat_hvp = torch.cat([product.contiguous().view(-1) for product in hessian_vector_product])
    return flat_hvp.data


def cg(f_Ax, b, cg_iters=10, residual_tol=1e-10):
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
    b (torch.Tensor):
    cg_iter (int): A integer, indicate the iteration for conjugate
        gradient.
    residual_tol (float): A float, indicate the terminate condition
        for the algorithm.

    Returns
    -------
    x (torch.Tensor): result of conjugate gradient
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros(b.size())
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

def line_search(f, x0, dx, expected_improvement, y0=None,
    backtrack_ratio=0.8, max_backtracks=15, accept_ratio=0.1, atol=1e-7):
    """
    Perform line search on the function f at x.

    Parameters
    ----------
    f (function): The function to perform line search on, it receive a tensor
        and return a float
    x0 (torch.Tensor): The current parameter value
    dx (torch.Tensor): The full descent direction. We shrink along this direction.
    expected_improvement (float): Expected amount of improvement when taking the
        full descent direction, typically computed by
        y0 - y ≈ (f_x|x=x0).dot(dx),
        where f_x|x=x0 is the gradient of f w.r.t. x, evaluated at x0.
    y0 (float): The initial value of f at x
    backtrack_ratio (float): Ratio to shrink the descent direction per line
        search step
    max_backtracks (int): Maximum number of backtracking steps
    accept_ratio (float): minimum acceptance ratio of
        actual_improvement / expected_improvement
    atol (float): Minimum expectd improvement to conduct line search

    Returns
    -------
    descent step (torch.Tensor):
    """
    if expected_improvement >= atol:
        if y0 is None:
            y0 = f(x0)
        for ratio in backtrack_ratio ** np.arange(max_backtracks):
            x = x0 - ratio * dx
            y = f(x)
            actual_improvement = y0 - y
            if actual_improvement / (expected_improvement * ratio) >= accept_ratio:
                logger.log('ExpectedImprovement: {}'.format(expected_improvement * ratio))
                logger.log('ActualImprovement: {}'.format(actual_improvement))
                logger.log('ImprovementRatio: {}'.format(actual_improvement /
                         (expected_improvement * ratio)))
            return x
    logger.log('ExpectedImprovement: {}'.format(expected_improvement))
    logger.log('ActualImprovement: {}'.format(0.0))
    logger.log('ImprovementRatio: {}'.format(0.0))
    return x0


class TRPO(BatchPolopt):
    """
    Trust region policy optimization.

    TRPO uses hessian free optimization.
    http://icml2010.haifa.il.ibm.com/papers/458.pdf
    """

    def __init__(
        self,
        step_size=0.01,
        damping=1e-8,
        use_line_search=True,
        use_finite_diff_hvp=False,
        symmetric_finite_diff=True,
        finite_diff_hvp_epsilon=1e-5,
        **kwargs
    ):
        """
        Trust region policy optimization method.

        Parameters
        ----------
        step_size (float): This is the constraint on the kl-divergence,
            that is D_KL(old||new) <= step_size
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
        super(TRPO, self).__init__(**kwargs)
        self.step_size = step_size
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
        surr_loss = surrogate_loss(self.policy, obs_var, action_var, adv_var, old_dist_var)
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
                surr_loss = surrogate_loss(self.policy, obs_var, action_var, adv_var, old_dist_var)
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
