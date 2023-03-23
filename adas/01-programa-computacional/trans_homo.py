import numpy as np
import sympy as sp
import mat_rot_gibbs as mrg


def num_trans_homo(
    e: np.ndarray, phi, _1Porg2x, _1Porg2y, _1Porg2z, _2Px, _2Py, _2Pz
) -> tuple[np.ndarray, np.ndarray]:
    """
    e: 3x1
    """

    mat_rot = mrg.num_mat_rot_gibbs(e, phi)
    aq = np.array([[_1Porg2x], [_1Porg2y], [_1Porg2z]])
    join_R_AQ = np.concatenate([mat_rot, aq], axis=1)
    th = np.concatenate([join_R_AQ, np.array([[0, 0, 0, 1]])], axis=0)
    p1 = np.array([[_2Px], [_2Py], [_2Pz], [1]])
    p2 = th @ p1
    return th, p2


def sym_trans_homo(
    e: sp.Matrix, phi, _1Porg2x, _1Porg2y, _1Porg2z, _2Px, _2Py, _2Pz
) -> tuple[sp.Matrix, sp.Matrix]:
    """
    e: 3x1
    """

    mat_rot = mrg.sym_mat_rot_gibbs(e, phi)
    aq = sp.Matrix([[_1Porg2x], [_1Porg2y], [_1Porg2z]])
    join_R_AQ = mat_rot.row_join(aq)
    th = join_R_AQ.col_join(sp.Matrix([[0, 0, 0, 1]]))
    p1 = sp.Matrix([[_2Px], [_2Py], [_2Pz], [1]])
    p2 = th @ p1
    return th, p2


def main():
    e = np.array([[1], [0], [0]])
    th, p2 = num_trans_homo(e, 30, 3, 10, -1, 0, -1, 2)
    print("Resultado numérico:")
    sp.pprint(th)
    sp.pprint(p2)

    ex, ey, ez = sp.symbols("ex ey ez")
    e = sp.Matrix([[ex], [ey], [ez]])
    th, p2 = sym_trans_homo(e, 30, 3, 10, -1, 0, -1, 2)
    print("Resultado simbólico:")
    sp.pprint(th)
    sp.pprint(p2)


if __name__ == "__main__":
    main()
