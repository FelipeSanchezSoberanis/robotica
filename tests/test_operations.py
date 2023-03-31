import unittest
import numpy as np
import src.numeric_operations as num_op


class TestOperations(unittest.TestCase):
    def test_mat_rot_gibbs(self):
        e = np.array([[0], [0], [1]])
        phi = 90

        calculated_mat_rot = num_op.mat_rot_gibbs(e, phi)
        expected_mat_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        np.testing.assert_almost_equal(calculated_mat_rot, expected_mat_rot)

    def test_mat_trans_homo(self):
        e = np.array([[1], [0], [0]])
        phi = 90
        vt = np.array([[8], [4], [12]])
        sf = 1
        vp = np.zeros((1, 3))

        calculated_mat_homo = num_op.mat_trans_homo(e, phi, vt, sf, vp)
        expected_mat_homo = np.array(
            [[1, 0, 0, 8], [0, 0, -1, 4], [0, 1, 0, 12], [0, 0, 0, 1]]
        )

        np.testing.assert_almost_equal(calculated_mat_homo, expected_mat_homo)
