import unittest
import operations.symbolic as sym_op
import operations.numeric as num_op
import sympy as sp
import numpy as np


class MainTests(unittest.TestCase):
    def test_rotar(self):
        ex_sym, ey_sym, ez_sym = sp.symbols("ex ey ez")
        ex_num, ey_num, ez_num = 15, 20, 35

        theta_sym = sp.symbols("theta")
        theta_num = 30

        e_sym = sp.Matrix([[ex_sym], [ey_sym], [ez_sym]])
        e_num = np.array([[ex_num], [ey_num], [ez_num]])

        sym_res = sym_op.rotate(e_sym, theta_sym)
        num_res = num_op.rotate(e_num, theta_num)

        sym_res_eval = sym_res.evalf(
            subs={
                "ex": ex_num,
                "ey": ey_num,
                "ez": ez_num,
                "theta": np.deg2rad(theta_num),
            }
        )
        sym_res_eval = np.array(sym_res_eval, dtype=float)

        np.testing.assert_allclose(sym_res_eval, num_res)

    def test_trotar(self):
        ex_sym, ey_sym, ez_sym = sp.symbols("ex ey ez")
        ex_num, ey_num, ez_num = 15, 20, 35

        theta_sym = sp.symbols("theta")
        theta_num = 30

        e_sym = sp.Matrix([[ex_sym, ey_sym, ez_sym]])
        e_num = np.array([[ex_num, ey_num, ez_num]])

        sym_res = sym_op.trotar(e_sym, theta_sym)
        num_res = num_op.trotar(e_num, theta_num)

        sym_res_eval = sym_res.evalf(
            subs={
                "ex": ex_num,
                "ey": ey_num,
                "ez": ez_num,
                "theta": np.deg2rad(theta_num),
            }
        )
        sym_res_eval = np.array(sym_res_eval, dtype=float)

        np.testing.assert_allclose(sym_res_eval, num_res)
