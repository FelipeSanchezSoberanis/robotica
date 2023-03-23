import numpy as np
import sympy as sp


def num_mat_rot_gibbs(e: np.ndarray, phi: float) -> np.ndarray:
    e = e.reshape(3, 1)
    phi = np.deg2rad(phi)

    ex, ey, ez = e[0, 0], e[1, 0], e[2, 0]
    se = np.array([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]], dtype=float)
    return e @ e.T + np.cos(phi) * (np.eye(3) - e @ e.T) + np.sin(phi) * se


def sym_mat_rot_gibbs(e: sp.Matrix, phi: sp.Symbol) -> sp.Matrix:
    e = e.reshape(3, 1)

    ex, ey, ez = e[0, 0], e[1, 0], e[2, 0]
    se = sp.Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])  # type: ignore
    return e @ e.T + sp.cos(phi) * (sp.eye(3) - e @ e.T) + sp.sin(phi) * se


def main():
    e = np.array([[0], [1], [0]])
    num = num_mat_rot_gibbs(e, 90)
    u = np.array([[2], [-1], [3]])
    print("Resultado numérico:")
    sp.pprint(num @ u)

    ex, ey, ez = sp.symbols("ex ey ez")
    phi = sp.symbols("phi")
    ux, uy, uz = sp.symbols("ux uy uz")
    e = sp.Matrix([[ex], [ey], [ez]])
    sym = sym_mat_rot_gibbs(e, phi)
    u = sp.Matrix([[ux], [uy], [uz]])
    print("Resultado simbólico:")
    sp.pprint(sym @ u)


if __name__ == "__main__":
    main()
