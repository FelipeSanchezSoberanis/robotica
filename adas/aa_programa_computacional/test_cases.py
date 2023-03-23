import unittest
import mat_rot_gibbs as mrg
import trans_homo as th
import trans_homo_inv as thi
import numpy as np


class TestAda01(unittest.TestCase):
    def test_mat_rot_gibbs_1(self):
        p = np.array([[1], [1], [2]])
        phi = 90
        axis = np.array([[0], [1], [0]])

        result = np.array([[2], [1], [-1]])

        rot_mat = mrg.num_mat_rot_gibbs(axis, phi)
        calculated_result = rot_mat @ p

        np.testing.assert_almost_equal(calculated_result, result)

    def test_trans_homo_1(self):
        e = np.array([[0], [0], [1]])
        phi = -90
        system = np.array([[0], [0], [0]])
        vector = np.array([[4], [8], [12]])

        result = np.array([[8], [-4], [12], [1]])

        calculated_result = th.num_trans_homo(e, phi, system, vector)

        np.testing.assert_almost_equal(calculated_result, result)

    def test_trans_homo_2(self):
        e = np.array([[0], [0], [0]])
        phi = 0
        system = np.array([[6], [-3], [8]])
        vector = np.array([[-2], [7], [3]])

        result = np.array([[4], [4], [11], [1]])

        calculated_result = th.num_trans_homo(e, phi, system, vector)

        np.testing.assert_almost_equal(calculated_result, result)

    def test_trans_homo_3(self):
        e = np.array([[1], [0], [0]])
        phi = 90
        system = np.array([[8], [-4], [12]])
        vector = np.array([[-3], [4], [-11]])

        result = np.array([[5], [7], [16], [1]])

        calculated_result = th.num_trans_homo(e, phi, system, vector)

        np.testing.assert_almost_equal(calculated_result, result)

    def test_trans_homo_inv_1(self):
        e = np.array([[0], [0], [1]])
        phi = 30
        system = np.array([[4], [3], [0]])

        result = np.array(
            [
                [0.866025403784439, 0.5, 0, -4.96410161513775],
                [-0.5, 0.866025403784439, 0, -0.598076211353316],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        calculated_result = thi.num_trans_homo_inv(e, phi, system)

        np.testing.assert_almost_equal(calculated_result, result)

    def test_complex(self):
        e = np.array([[0], [0], [1]])
        phi = 90
        aq = np.array([[0], [0.4], [0.2]])

        t_1_2 = th.num_trans_homo_th(e, phi, aq)

        t_3_1 = np.array([[0, 1, 0, -0.5], [1, 0, 0, 0.5], [0, 0, -1, 2], [0, 0, 0, 1]])

        calculated_result = t_3_1 @ t_1_2

        result = np.array(
            [[1, 0, 0, -0.1], [0, -1, 0, 0.5], [0, 0, -1, 1.8], [0, 0, 0, 1]]
        )

        np.testing.assert_almost_equal(calculated_result, result)


if __name__ == "__main__":
    unittest.main()
