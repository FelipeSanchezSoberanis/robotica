import numpy as np
import sympy as sp
import mat_rot_gibbs as mrg


def num_trans_homo(
    e: np.ndarray,
    phi: float,
    _1Porg2x: float,
    _1Porg2y: float,
    _1Porg2z: float,
    _2Px: float,
    _2Py: float,
    _2Pz: float,
) -> tuple[np.ndarray, np.ndarray]:
    mat_rot = mrg.num_mat_rot_gibbs(e, phi)
    aq = np.array([[_1Porg2x], [_1Porg2y], [_1Porg2z]])
    join_R_AQ = np.concatenate([mat_rot, aq], axis=1)
    th = np.concatenate([join_R_AQ, np.array([[0, 0, 0, 1]])], axis=0)
    p1 = np.array([[_2Px], [_2Py], [_2Pz], [1]])
    p2 = th @ p1
    return th, p2


def sym_trans_homo(
    e: sp.Matrix,
    phi: sp.Symbol,
    _1Porg2x: sp.Symbol,
    _1Porg2y: sp.Symbol,
    _1Porg2z: sp.Symbol,
    _2Px: sp.Symbol,
    _2Py: sp.Symbol,
    _2Pz: sp.Symbol,
) -> tuple[sp.Matrix, sp.Matrix]:
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
    phi = sp.symbols("phi")

    _1Porg2x, _1Porg2y, _1Porg2z, _2Px, _2Py, _2Pz = sp.symbols(
        "_1Porg2x _1Porg2y _1Porg2z _2Px _2Py _2Pz"
    )

    th, p2 = sym_trans_homo(e, phi, _1Porg2x, _1Porg2y, _1Porg2z, _2Px, _2Py, _2Pz)
    print("Resultado simbólico:")
    sp.pprint(th)
    sp.pprint(p2)


if __name__ == "__main__":
    main()
