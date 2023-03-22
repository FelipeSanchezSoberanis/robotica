import sympy as sp
import numpy as np


def num_mat_rot_gibbs(ex: float, ey: float, ez: float, phi: float) -> np.ndarray:
    phi = np.deg2rad(phi)
    e = np.array([ex, ey, ez])
    se: np.ndarray = np.array([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])  # type: ignore
    return e * e.T + np.cos(phi) * (np.eye(3) - e * e.T) + np.sin(phi) * se


def sym_mat_rot_gibbs(
    ex: sp.Symbol, ey: sp.Symbol, ez: sp.Symbol, phi: sp.Symbol
) -> sp.Matrix:
    e = sp.Matrix([ex, ey, ez])
    se = sp.Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])
    return e * e.T + sp.cos(phi) * (sp.eye(3) - e * e.T) + sp.sin(phi) * se


def main():
    num = num_mat_rot_gibbs(0, 0, 1, 30)
    print("Resultado numérico:")
    sp.pprint(num)

    ex, ey, ez, phi = sp.symbols("ex ey ez phi")
    sym = sym_mat_rot_gibbs(ex, ey, ez, phi)
    print("Resultado simbólico:")
    sp.pprint(sym)


if __name__ == "__main__":
    main()
