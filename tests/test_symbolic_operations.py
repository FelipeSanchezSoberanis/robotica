import unittest
import numpy as np
import sympy as sp
import src.numeric_operations as num_op
import src.symbolic_operations as sym_op
import random


class TestSymbolicOperations(unittest.TestCase):
    def test_mat_rot_gibbs(self):
        #  Symbolic operation
        ex, ey, ez = sp.symbols("ex ey ez")
        e = sp.Matrix([[ex], [ey], [ez]])
        phi = sp.symbols("phi")

        sym_res = sym_op.mat_rot_gibbs(e, phi)

        #  Numeric operation
        e = np.random.randn(3, 1)
        phi = random.randint(0, 360)

        num_res = num_op.mat_rot_gibbs(e, phi)

        #  Comparison
        sym_subs = {"ex": e[0, 0], "ey": e[1, 0], "ez": e[2, 0], "phi": np.deg2rad(phi)}
        sym_eval = np.array(sym_res.evalf(subs=sym_subs), dtype=float)

        np.testing.assert_almost_equal(num_res, sym_eval)

    def test_mat_trans_homo(self):
        #  Symbolic operation
        ex, ey, ez = sp.symbols("ex ey ez")
        e = sp.Matrix([[ex], [ey], [ez]])
        phi = sp.symbols("phi")
        vtx, vty, vtz = sp.symbols("vtx vty vtz")
        vt = sp.Matrix([[vtx], [vty], [vtz]])
        sf = sp.symbols("sf")
        vpx, vpy, vpz = sp.symbols("vpx vpy vpz")
        vp = sp.Matrix([[vpx], [vpy], [vpz]])

        sym_res = sym_op.mat_trans_homo(e, phi, vt, sf, vp)

        #  Numeric operation
        e = np.random.randn(3, 1)
        phi = random.randint(0, 360)
        vt = np.random.randn(3, 1)
        sf = random.randint(-1_000_000, 1_000_000)
        vp = np.random.randn(1, 3)

        num_res = num_op.mat_trans_homo(e, phi, vt, sf, vp)

        #  Comparison
        sym_subs = {
            "ex": e[0, 0],
            "ey": e[1, 0],
            "ez": e[2, 0],
            "phi": np.deg2rad(phi),
            "vtx": vt[0, 0],
            "vty": vt[1, 0],
            "vtz": vt[2, 0],
            "sf": sf,
            "vpx": vp[0, 0],
            "vpy": vp[0, 1],
            "vpz": vp[0, 2],
        }
        sym_eval = np.array(sym_res.evalf(subs=sym_subs), dtype=float)

        np.testing.assert_almost_equal(num_res, sym_eval)

    def test_mat_trans_homo_inv(self):
        #  Symbolic operation
        ex, ey, ez = sp.symbols("ex ey ez")
        e = sp.Matrix([[ex], [ey], [ez]])
        phi = sp.symbols("phi")
        vtx, vty, vtz = sp.symbols("vtx vty vtz")
        vt = sp.Matrix([[vtx], [vty], [vtz]])
        sf = sp.symbols("sf")
        vpx, vpy, vpz = sp.symbols("vpx vpy vpz")
        vp = sp.Matrix([[vpx], [vpy], [vpz]])

        sym_res = sym_op.mat_trans_homo_inv(e, phi, vt, sf, vp)

        #  Numeric operation
        e = np.random.randn(3, 1)
        phi = random.randint(0, 360)
        vt = np.random.randn(3, 1)
        sf = random.randint(-1_000_000, 1_000_000)
        vp = np.random.randn(1, 3)

        num_res = num_op.mat_trans_homo_inv(e, phi, vt, sf, vp)

        #  Comparison
        sym_subs = {
            "ex": e[0, 0],
            "ey": e[1, 0],
            "ez": e[2, 0],
            "phi": np.deg2rad(phi),
            "vtx": vt[0, 0],
            "vty": vt[1, 0],
            "vtz": vt[2, 0],
            "sf": sf,
            "vpx": vp[0, 0],
            "vpy": vp[0, 1],
            "vpz": vp[0, 2],
        }
        sym_eval = np.array(sym_res.evalf(subs=sym_subs), dtype=float)

        np.testing.assert_almost_equal(num_res, sym_eval)
