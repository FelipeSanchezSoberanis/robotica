import unittest
import numpy as np
import src.numeric_operations as num_op


class TestNumericOperations(unittest.TestCase):
    def test_mat_rot_gibbs(self):
        rand_matrix = np.random.randn(3, 1)
        print(rand_matrix)

    def test_mat_trans_homo(self):
        pass

    def test_mat_trans_homo_inv(self):
        pass
