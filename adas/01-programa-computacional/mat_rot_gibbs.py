import sympy as sp
import numpy as np


def num_mat_rot_gibbs(u: np.ndarray, phi: float, e: np.ndarray) -> np.ndarray:
    u = u.reshape(3, 1)
    e = e.reshape(3, 1)

    ang = np.deg2rad(phi)
    uni = e / np.linalg.norm(e)
    se = np.array(
        [
            np.array([0, -uni[2, 0], uni[1, 0]]),
            np.array([uni[2, 0], 0, -uni[0, 0]]),
            np.array([-uni[1, 0], uni[0, 0], 0]),
        ]
    )
    q = (
        np.outer(uni, uni)
        + (np.identity(3) - np.outer(uni, uni)) * np.cos(ang)
        + se * np.sin(ang)
    )
    return q @ u


def sym_mat_rot_gibbs(
    ex: sp.Symbol, ey: sp.Symbol, ez: sp.Symbol, phi: sp.Symbol
) -> sp.Matrix:
    e = sp.Matrix([ex, ey, ez])
    se = sp.Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])
    return e * e.T + sp.cos(phi) * (sp.eye(3) - e * e.T) + sp.sin(phi) * se


def main():
    u = np.array([[10], [2], [-5]])
    phi = 30
    e = np.array([[1 / np.sqrt(3)], [1 / np.sqrt(3)], [-1 / np.sqrt(3)]])
    num = num_mat_rot_gibbs(u, phi, e)
    print("Resultado numérico:")
    sp.pprint(num)


if __name__ == "__main__":
    main()
