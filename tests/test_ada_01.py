import unittest
import numpy as np
import src.numeric_operations as num_op


class TestAda01(unittest.TestCase):
    def test_mat_rot_gibbs_1(self):
        p = np.array([[1], [1], [2]])
        phi = 90
        axis = np.array([[0], [1], [0]])

        result = np.array([[2], [1], [-1]])

        rot_mat = num_op.mat_rot_gibbs(axis, phi)
        calculated_result = rot_mat @ p

        np.testing.assert_almost_equal(calculated_result, result)

    def test_trans_homo_1(self):
        e = np.array([[0], [0], [1]])
        phi = -90
        vt = np.array([[0], [0], [0]])
        sf = 1
        vp = np.zeros((1, 3))

        p = np.array([[4], [8], [12]])
        mat_homo = num_op.mat_trans_homo(e, phi, vt, sf, vp)

        result = np.array([[8], [-4], [12], [1]])
        calculated_result = num_op.apply_homo_trans(mat_homo, p)

        np.testing.assert_almost_equal(calculated_result, result)

    def test_trans_homo_2(self):
        e = np.array([[0], [0], [0]])
        phi = 0
        vt = np.array([[6], [-3], [8]])
        sf = 1
        vp = np.zeros((1, 3))

        p = np.array([[-2], [7], [3]])
        mat_homo = num_op.mat_trans_homo(e, phi, vt, sf, vp)

        result = np.array([[4], [4], [11], [1]])
        calculated_result = num_op.apply_homo_trans(mat_homo, p)

        np.testing.assert_almost_equal(calculated_result, result)

    def test_trans_homo_3(self):
        e = np.array([[1], [0], [0]])
        phi = 90
        vt = np.array([[8], [-4], [12]])
        sf = 1
        vp = np.zeros((1, 3))

        p = np.array([[-3], [4], [-11]])
        mat_homo = num_op.mat_trans_homo(e, phi, vt, sf, vp)

        result = np.array([[5], [7], [16], [1]])
        calculated_result = num_op.apply_homo_trans(mat_homo, p)

        np.testing.assert_almost_equal(calculated_result, result)

    def test_trans_homo_inv_1(self):
        e = np.array([[0], [0], [1]])
        phi = 30
        vt = np.array([[4], [3], [0]])
        sf = 1
        vp = np.zeros((1, 3))

        result = np.array(
            [
                [0.866025403784439, 0.5, 0, -4.96410161513775],
                [-0.5, 0.866025403784439, 0, -0.598076211353316],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        calculated_result = num_op.mat_trans_homo_inv(e, phi, vt, sf, vp)

        np.testing.assert_almost_equal(calculated_result, result)

    def test_mult_rot_trans(self):
        e = np.array([[0], [0], [1]])
        phi = 90
        vt = np.array([[0], [0.4], [0.2]])
        sf = 1
        vp = np.zeros((1, 3))

        calculated_result = num_op.mat_trans_homo(e, phi, vt, sf, vp)

        cam_e_1 = np.array([[0], [0], [1]])
        cam_phi_1 = 90
        cam_tran_1 = np.array([[-0.5], [0.5], [2]])

        cam_e_2 = np.array([[1], [0], [0]])
        cam_phi_2 = 180
        cam_tran_2 = np.array([[0], [0], [0]])

        cam_es = [cam_e_1, cam_e_2]
        cam_phis = [cam_phi_1, cam_phi_2]
        cam_trans = [cam_tran_1, cam_tran_2]

        mat_homo = np.eye(4)
        for e, phi, cam_tran in zip(cam_es, cam_phis, cam_trans):
            mat_homo = mat_homo @ num_op.mat_trans_homo(e, phi, cam_tran, sf, vp)

        calculated_result = mat_homo @ calculated_result

        result = np.array(
            [[1, 0, 0, -0.1], [0, -1, 0, 0.5], [0, 0, -1, 1.8], [0, 0, 0, 1]]
        )

        np.testing.assert_almost_equal(calculated_result, result)
