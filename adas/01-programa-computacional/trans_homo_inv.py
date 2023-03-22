import numpy as np
import sympy as sp


def num_mat_rot_inv(ex: float, ey: float, ez: float, phi: float) -> np.ndarray:
    phi = np.deg2rad(phi)
    e = np.array([ex, ey, ez])
    se: np.array = np.array([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])  # type: ignore
    rot: np.ndarray = e * e.T + np.cos(phi) * (np.eye(3) - e * e.T) + np.sin(phi) * se
    return rot.T


def sym_mat_rot_inv(
    ex: sp.Symbol, ey: sp.Symbol, ez: sp.Symbol, phi: sp.Symbol
) -> sp.Matrix:
    e = sp.Matrix([ex, ey, ez])
    se = sp.Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])
    rot: sp.Matrix = e * e.T + sp.cos(phi) * (sp.eye(3) - e * e.T) + sp.sin(phi) * se
    return rot.T


def num_trans_homo_inv(
    ex: float,
    ey: float,
    ez: float,
    phi: float,
    _1Porg2x: float,
    _1Porg2y: float,
    _1Porg2z: float,
) -> np.ndarray:
    mat_rot_inv = num_mat_rot_inv(ex, ey, ez, phi)
    aq = np.array([[_1Porg2x], [_1Porg2y], [_1Porg2z]])
    raq = -mat_rot_inv @ aq
    join_R_AQ = np.concatenate([mat_rot_inv, raq], axis=1)
    return np.concatenate([join_R_AQ, np.array([[0, 0, 0, 1]])], axis=0)


def sym_trans_homo_inv(
    ex: sp.Symbol,
    ey: sp.Symbol,
    ez: sp.Symbol,
    phi: sp.Symbol,
    _1Porg2x: sp.Symbol,
    _1Porg2y: sp.Symbol,
    _1Porg2z: sp.Symbol,
) -> sp.Matrix:
    mat_rot_inv = sym_mat_rot_inv(ex, ey, ez, phi)
    aq = sp.Matrix([[_1Porg2x], [_1Porg2y], [_1Porg2z]])
    raq: sp.Matrix = -mat_rot_inv @ aq
    join_R_AQ: sp.Matrix = mat_rot_inv.row_join(raq)
    return join_R_AQ.col_join(sp.Matrix([[0, 0, 0, 1]]))


def main():
    num = num_trans_homo_inv(0, 0, 1, 30, 4, 3, 0)
    print("Resultado numérico:")
    sp.pprint(num)

    ex, ey, ez, phi, _1Porg2x, _1Porg2y, _1Porg2z = sp.symbols(
        "ex ey ez phi _1Porg2x _1Porg2y _1Porg2z"
    )
    sym = sym_trans_homo_inv(ex, ey, ez, phi, _1Porg2x, _1Porg2y, _1Porg2z)
    print("Resultado simbólico:")
    sp.pprint(sym)


if __name__ == "__main__":
    main()
