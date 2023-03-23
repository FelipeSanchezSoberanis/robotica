import unittest
import adas.aa_programa_computacional.mat_rot_gibbs as mrg
import adas.aa_programa_computacional.trans_homo as th
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

    def test_mat_rot_gibbs_2(self):
        p = np.array([[1], [2], [3]])
        phi = 45
        axis = np.array([[0], [0], [0]])

        result = np.array([[2.828], [2], [1.414]])

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
