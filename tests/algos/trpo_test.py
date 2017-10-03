import unittest

import numpy as np
import torch

from pytorchrl.algos.trpo import cg

class TestTRPO(unittest.TestCase):

    def test_cg(self):
        A = torch.randn(5, 5)
        A = A.t().mm(A)
        b = torch.randn(5)
        x = cg(lambda x: A.matmul(x), b, cg_iters=5)
        assert np.allclose(A.matmul(x).numpy(), b.numpy())

if __name__ == '__main__':
    unittest.main()