import numpy as np
import sympy as sp


def num_mat_rot_inv(e: np.ndarray, phi: float) -> np.ndarray:
    _phi: float = np.deg2rad(phi)
    ex, ey, ez = (
        e[0, 0],
        e[1, 0],
        e[2, 0],
    )
    se = np.array([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])  # type: ignore
    mat_rot: np.ndarray = (
        e @ e.T + np.cos(_phi) * (np.eye(3) - e @ e.T) + np.sin(_phi) * se
    )
    return mat_rot.T


def num_trans_homo_inv(e: np.ndarray, phi: float, aq: np.ndarray) -> np.ndarray:
    rot_inv = num_mat_rot_inv(e, phi)
    raq = -rot_inv @ aq
    join_R_AQ = np.concatenate([rot_inv, raq], axis=1)
    thi = np.concatenate([join_R_AQ, np.array([[0, 0, 0, 1]])], axis=0)
    return thi


def sym_mat_rot_inv(e: sp.Matrix, phi: sp.Symbol) -> sp.Matrix:
    ex, ey, ez = (
        e[0, 0],
        e[1, 0],
        e[2, 0],
    )
    se = sp.Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])  # type: ignore
    mat_rot: sp.Matrix = (
        e @ e.T + sp.cos(phi) * (sp.eye(3) - e @ e.T) + sp.sin(phi) * se
    )
    return mat_rot.T


def sym_trans_homo_inv(e: sp.Matrix, phi: sp.Symbol, aq: sp.Matrix) -> sp.Matrix:
    rot_inv = sym_mat_rot_inv(e, phi)
    raq: sp.Matrix = -rot_inv @ aq
    join_R_AQ: sp.Matrix = rot_inv.row_join(raq)
    thi: sp.Matrix = join_R_AQ.col_join(sp.Matrix([[0, 0, 0, 1]]))
    return thi


def main():
    #  num = num_trans_homo_inv(1, 0, 0, 30, 3, 10, -1)
    #  print("Resultado numérico:")
    #  sp.pprint(num)

    #  ex, ey, ez, phi, _1Porg2x, _1Porg2y, _1Porg2z = sp.symbols(
    #      "ex, ey, ez, phi, _1Porg2x, _1Porg2y, _1Porg2z"
    #  )
    #  sym = sym_trans_homo_inv(ex, ey, ez, phi, _1Porg2x, _1Porg2y, _1Porg2z)
    #  print("Resultado simbólico:")
    #  sp.pprint(sym)
    pass


if __name__ == "__main__":
    main()
