import numpy as np
import sympy as sp
import mat_rot_gibbs as mrg


def num_trans_homo(
    ex: float,
    ey: float,
    ez: float,
    phi: float,
    _1Porg2x: float,
    _1Porg2y: float,
    _1Porg2z: float,
    _2Px: float,
    _2Py: float,
    _2Pz: float,
) -> np.ndarray:
    mat_rot = mrg.num_mat_rot_gibbs(np.array([[ex], [ey], [ez]]), phi)
    aq = np.array([[_1Porg2x], [_1Porg2y], [_1Porg2z]])
    join_r_aq = np.concatenate([mat_rot, aq], axis=1)
    th = np.concatenate([join_r_aq, np.array([[0, 0, 0, 1]])], axis=0)
    p1 = np.array([[_2Px], [_2Py], [_2Pz], [1]])
    return th @ p1


def sym_trans_homo(
    ex: sp.Symbol,
    ey: sp.Symbol,
    ez: sp.Symbol,
    phi: sp.Symbol,
    _1Porg2x: sp.Symbol,
    _1Porg2y: sp.Symbol,
    _1Porg2z: sp.Symbol,
    _2Px: sp.Symbol,
    _2Py: sp.Symbol,
    _2Pz: sp.Symbol,
) -> sp.Matrix:
    mat_rot = mrg.sym_mat_rot_gibbs(sp.Matrix([[ex], [ey], [ez]]), phi)
    aq = sp.Matrix([[_1Porg2x], [_1Porg2y], [_1Porg2z]])
    join_r_aq: sp.Matrix = mat_rot.row_join(aq)
    th: sp.Matrix = join_r_aq.col_join(sp.Matrix([[0, 0, 0, 1]]))
    p1 = sp.Matrix([[_2Px], [_2Py], [_2Pz], [1]])
    return th @ p1


def main():
    num = num_trans_homo(0, 0, 1, 30, 10, 5, 0, 3, 7, 0)
    print("Resultado numérico:")
    sp.pprint(num)

    ex, ey, ez, phi, _1Porg2x, _1Porg2y, _1Porg2z, _2Px, _2Py, _2Pz = sp.symbols(
        "ex ey ez phi _1Porg2x _1Porg2y _1Porg2z _2Px _2Py _2Pz"
    )
    sym = sym_trans_homo(
        ex, ey, ez, phi, _1Porg2x, _1Porg2y, _1Porg2z, _2Px, _2Py, _2Pz
    )
    print("Resultado simbólico:")
    sp.pprint(sym)


if __name__ == "__main__":
    main()
