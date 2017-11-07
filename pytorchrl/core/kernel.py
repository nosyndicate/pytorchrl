import numpy as np
import unittest
import torch
from torch.autograd import Variable

import sys

class Kernel(object):
    """
    Kernel Tensor

    The output is a tensor of shape ... * Kx * Ky, where
    the dimensions Kx and Ky are determined by the inputs xs and ys.

    For example,
    shape of xs: ... * Kx * D
    shape of ys: ... * Ky * D

    output shape: ... * Kx * Ky
    grad shape: ... * Kx * Ky * D

    where the leading dimensions ... needs to match. The gradients is taken
    with respect to the first input ('xs')

    This is used to compute the kernel k(z_j, z_i)
    and its gradient \nabla_{z_j} k(z_j, z_i)

    in SVGD

    phi(z_i) = 1/n \sum_{j=1}^{n} [\nabla_{z_j} log p(z_j) k(z_j, z_i) +
                                    \nabla_{z_j} k(z_j, z_i)]


    When we use the kernel in Soft Q-Learning, the leading dimension ...
    is usually N which is the batch size and the last dimension D is the
    number of dimension of actions.
    """

    def __init__(self, xs, ys, kappa, grad_x):
        """
        Parameters
        ----------
        xs (Variable): Left particles.
        ys (Variable): Right particles
        kappa (Variable): Kernel value
        grad_x (Variable): Gradient of 'kappa' with respect to 'xs'
        """

        self.kappa = kappa  # ... x Kx x Ky
        self.xs = xs  # ... x Kx x D
        self.ys = ys  # ... x Ky x D
        self.grad_x = grad_x  # ... x Kx x Ky x D

    @property
    def grad(self):
        """
        Returns the gradient of the Kernel matrix with respect to the first
        argument of the kernel function.
        """
        return self.grad_x  # ... x Kx x Ky x D


class AdaptiveIsotropicGaussianKernel(Kernel):
    def __init__(self, xs, ys, h_min=1e-3):
        """
        Gaussian kernel with dynamic bandwidth equal to
        median_distance / log(Kx).

        Parameters
        ----------
        xs (Variable): Left particles.
        ys (Variable): Right particles.
        h_min (float): Minimum bandwidth.
        """
        self.h_min = h_min

        Kx, D = xs.size()[-2:]
        Ky, D2 = ys.size()[-2:]
        assert D == D2

        leading_shape = xs.size()[:-2]

        # Compute the pairwise distances of left and right particles.
        diff = xs.unsqueeze(-2) - ys.unsqueeze(-3)  # ... * Kx * Ky * D
        dist_sq = diff.pow(2).sum(dim=-1, keepdim=False) # ... * Kx * Ky

        # Get median.
        shape = leading_shape + torch.Size([Kx * Ky])
        values, _ = dist_sq.view(shape).topk(
            k=(Kx * Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
        ) # ... * floor(Ks*Kd/2)

        medians_sq = values[..., -1]  # ... (shape) (last element is the median)

        h = medians_sq / np.log(Kx)  # ... (shape)
        h = torch.max(h, Variable(torch.FloatTensor([self.h_min])))
        h = h.detach() # Just in case.
        h_expanded_twice = h.unsqueeze(-1).unsqueeze(-1) # ... * 1 * 1

        kappa = torch.exp(- dist_sq / h_expanded_twice)  # ... x Kx x Ky

        # Construct the gradient
        h_expanded_thrice = h_expanded_twice.unsqueeze(-1)  # ... x 1 x 1 x 1

        kappa_expanded = kappa.unsqueeze(-1)  # ... x Kx x Ky x 1

        kappa_grad = - 2 * diff / h_expanded_thrice * kappa_expanded # ... x Kx x Ky x D

        super().__init__(xs, ys, kappa, kappa_grad)

        self.db = dict()
        self.db['medians_sq'] = medians_sq



class TestAdaptiveIsotropicGaussianKernel(unittest.TestCase):
    def test_multi_axis(self):
        Kx, Ky = 3, 5
        N = 1
        M = 10
        D = 20
        h_min = 1E-3

        x = np.random.randn(N, M, Kx, D)
        y = np.random.randn(N, M, Ky, D)
        x_var = Variable(torch.from_numpy(x).type(torch.FloatTensor))
        y_var = Variable(torch.from_numpy(y).type(torch.FloatTensor))

        kernel = AdaptiveIsotropicGaussianKernel(x_var, y_var, h_min)

        medians_sq = kernel.db['medians_sq'].data.numpy()
        kappa, kappa_grad = kernel.kappa.data.numpy(), kernel.grad.data.numpy()

        # Compute and test the median with numpy.
        diff = x[..., None, :] - y[..., None, :, :]

        # noinspection PyTypeChecker
        dist_sq = np.sum(diff**2, axis=-1)
        medians_sq_np = np.median(dist_sq.reshape((N, M, -1)), axis=-1)

        np.testing.assert_almost_equal(medians_sq, medians_sq_np,
                                       decimal=5, err_msg='Wrong median.')
        # Test the kernel matrix.
        h = medians_sq_np / np.log(Kx)
        h = np.maximum(h, h_min)

        kappa_np = np.exp(- dist_sq / h[..., None, None])

        np.testing.assert_almost_equal(kappa, np.float32(kappa_np),
                                       decimal=5, err_msg='Wrong kernel.')

        # Test the gradient.
        grad_np = - 2*diff/h[..., None, None, None]*kappa_np[..., None]

        # noinspection PyTypeChecker
        np.testing.assert_almost_equal(grad_np, kappa_grad)


    def test_trivial(self):
        """
        Test a trivial case of single particle, which can be easily
        confirmed without tensor algebra.
        """
        Kx, Ky = 2, 1  # Kx need to be at least 2
        N = 1
        D = 1
        h_min = 1E-3

        x = Variable(torch.from_numpy(np.zeros((N, Kx, D))).type(
                torch.FloatTensor))
        y = Variable(torch.from_numpy(np.ones((N, Ky, D))).type(
                torch.FloatTensor))

        kernel = AdaptiveIsotropicGaussianKernel(x, y, h_min)

        medians_sq = kernel.db['medians_sq'].data.numpy()
        kappa, kappa_grad = kernel.kappa.data.numpy(), kernel.grad.data.numpy()

        # Test median.
        self.assertEqual(medians_sq[0], 1, msg='Wrong median.')

        # Test kappa.
        np.testing.assert_equal(kappa.squeeze(), np.exp(-np.log([Kx, Kx])),
            err_msg='Wrong kernel.')

        # Test grad_x kappa.
        grad_np = 2 * np.log(Kx) * np.exp(-np.log([Kx, Kx]))

        # noinspection PyTypeChecker
        np.testing.assert_equal(kappa_grad.squeeze(), np.float32(np.squeeze(grad_np)))



if __name__ == '__main__':
    unittest.main()