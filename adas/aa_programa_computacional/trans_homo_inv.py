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
    num_e = np.array([[0], [0], [1]])
    num_phi = 30
    num_p = np.array([[4], [3], [0]])

    num_res = num_trans_homo_inv(num_e, num_phi, num_p)
    print("Resultado numérico:")
    sp.pprint(num_res)

    ex, ey, ez, phi, px, py, pz = sp.symbols("ex ey ez phi px py pz ")
    sym_e = sp.Matrix([[ex], [ey], [ez]])
    sym_p = sp.Matrix([[px], [py], [pz]])

    sym_res = sym_trans_homo_inv(sym_e, phi, sym_p)
    print("Resultado simbólico:")
    sp.pprint(sym_res)


if __name__ == "__main__":
    main()
