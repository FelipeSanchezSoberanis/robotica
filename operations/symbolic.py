import sympy as sp


def rotate(e: sp.Matrix, theta: sp.Symbol) -> sp.Matrix:
    ex: sp.Symbol = e[0, 0]  # type: ignore
    ey: sp.Symbol = e[1, 0]  # type: ignore
    ez: sp.Symbol = e[2, 0]  # type: ignore

    se = sp.Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])

    return e @ e.T + (sp.eye(3) - e @ e.T) * sp.cos(theta) + se * sp.sin(theta)


