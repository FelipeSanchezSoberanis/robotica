import unittest
import numpy as np
import src.numeric_operations as num_op
import random


class TestNumericOperations(unittest.TestCase):
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

    def test_mat_trans_homo_inv(self):
        e = np.array([[0], [0], [1]])
        phi = 30
        vt = np.array([[4], [3], [0]])
        sf = 1
        vp = np.zeros((1, 3))

        calculated_mat_homo_inv = num_op.mat_trans_homo_inv(e, phi, vt, sf, vp)
        expected_mat_homo_inv = np.array(
            [
                [0.8660254, 0.5, 0, -4.96410162],
                [-0.5, 0.8660254, 0, -0.59807621],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        np.testing.assert_almost_equal(calculated_mat_homo_inv, expected_mat_homo_inv)

    def test_transformation_chaining(self):
        e = np.array([[0], [0], [1]])
        phi = 30
        vt = np.array([[4], [3], [0]])
        sf = 1
        vp = np.zeros((1, 3))

        rand_int = lambda: random.randint(-1_000_000, 1_000_000)

        p_original = np.array([[rand_int()], [rand_int()], [rand_int()], [1]])

        mat_trans = num_op.mat_trans_homo(e, phi, vt, sf, vp)
        mat_trans_inv = num_op.mat_trans_homo_inv(e, phi, vt, sf, vp)

        p_translated = mat_trans @ p_original

        np.testing.assert_almost_equal(mat_trans_inv @ p_translated, p_original)
