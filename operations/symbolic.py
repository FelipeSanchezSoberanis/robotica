import sympy as sp
import numpy as np


def rotate(e: sp.Matrix, theta: sp.Symbol) -> sp.Matrix:
    ex: sp.Symbol = e[0, 0]  # type: ignore
    ey: sp.Symbol = e[1, 0]  # type: ignore
    ez: sp.Symbol = e[2, 0]  # type: ignore

    se = sp.Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])

    return e @ e.T + (sp.eye(3) - e @ e.T) * sp.cos(theta) + se * sp.sin(theta)


def trotar(e: sp.Matrix, theta: sp.Symbol) -> sp.Matrix:
    ex: sp.Symbol = e[0, 0]  # type: ignore
    ey: sp.Symbol = e[0, 1]  # type: ignore
    ez: sp.Symbol = e[0, 2]  # type: ignore

    se = sp.Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])

    parte1: sp.Matrix = (
        e.T @ e + sp.cos(theta) * (sp.eye(3) - e.T @ e) + sp.sin(theta) * se
    )
    parte2 = parte1.row_join(sp.Matrix([[0], [0], [0]]))
    res = parte2.col_join(sp.Matrix([[0, 0, 0, 1]]))

    return res
