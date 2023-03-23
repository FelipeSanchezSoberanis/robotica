import numpy as np
import sympy as sp
import mat_rot_gibbs as mrg


def num_trans_homo(
    e: np.ndarray,
    phi: float,
    aq: np.ndarray,
    p1: np.ndarray,
) -> np.ndarray:
    e = e.reshape(3, 1)
    aq = aq.reshape(3, 1)
    p1 = p1.reshape(3, 1)

    mat_rot = mrg.num_mat_rot_gibbs(e, phi)
    join_R_AQ = np.concatenate([mat_rot, aq], axis=1)
    th = np.concatenate([join_R_AQ, np.array([[0, 0, 0, 1]])], axis=0)
    p1 = np.concatenate([p1, np.array([[1]])], axis=0)
    p2 = th @ p1
    return p2


def sym_trans_homo(
    e: sp.Matrix, phi: sp.Symbol, aq: sp.Matrix, p1: sp.Matrix
) -> sp.Matrix:
    e = e.reshape(3, 1)
    aq = aq.reshape(3, 1)
    p1 = p1.reshape(3, 1)

    mat_rot = mrg.sym_mat_rot_gibbs(e, phi)
    join_R_AQ: sp.Matrix = mat_rot.row_join(aq)
    th: sp.Matrix = join_R_AQ.col_join(sp.Matrix([[0, 0, 0, 1]]))
    p1 = p1.col_join(sp.Matrix([[1]]))
    p2: sp.Matrix = th @ p1
    return p2


def main():
    e = np.array([[0], [0], [1]])
    phi = -90
    system = np.array([[0], [0], [0]])
    vector = np.array([[4], [8], [12]])
    result = num_trans_homo(e, phi, system, vector)
    print("Resultado numérico:")
    sp.pprint(result)

    ex, ey, ez = sp.symbols("ex ey ez")
    vx, vy, vz = sp.symbols("vx vy vz")
    sx, sy, sz = sp.symbols("sx sy sz")
    phi = sp.symbols("phi")
    e = sp.Matrix([[ex], [ey], [ez]])
    system = sp.Matrix([[sx], [sy], [sz]])
    vector = sp.Matrix([[vx], [vy], [vz]])
    result = sym_trans_homo(e, phi, system, vector)
    print("Resultado simbólico:")
    sp.pprint(result)


if __name__ == "__main__":
    main()
