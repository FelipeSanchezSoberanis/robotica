import sympy as sp


def mat_rot_gibbs(e: sp.Matrix, phi: sp.Symbol) -> sp.Matrix:
    """
    Calculates the Gibbs rotation matrix when rotating "phi" degrees around the
    axis "e".

    Parameters:
    - e (sp.Matrix): Axis of rotation. Can be of shape (1, 3) or (3, 1).
    - phi (sp.Symbol): Degrees of rotation. Has to be in degrees, not radians.

    Returns:
    - sp.Matrix: Gibbs rotation matrix. Has shape (3, 3).
    """
    e = e.reshape(3, 1)

    ex, ey, ez = e[0, 0], e[1, 0], e[2, 0]
    se = sp.Matrix([[0, -1 * ez, ey], [ez, 0, -1 * ex], [-1 * ey, ex, 0]])
    mat_rot = e @ e.T + sp.cos(phi) * (sp.eye(3) - e @ e.T) + sp.sin(phi) * se

    return mat_rot


def mat_trans_homo(
    e: sp.Matrix, phi: sp.Symbol, vt: sp.Matrix, sf: sp.Symbol, vp: sp.Matrix
) -> sp.Matrix:
    """
    Calculates the homogenous transformation matrix given an axis of rotation,
    degree of rotation, vector of translation, scaling factor and vector of
    perspective.

    Parameters:
    - e (sp.Matrix): Axis of rotation. Can be of shape (1, 3) or (3, 1).
    - phi (sp.Symbol): Degrees of rotation. Has to be in degrees, not radians.
    - vt (sp.Matrix): Translation vector. Can be of shape (1, 3) or (3, 1).
    - sf (sp.Symbol): Scaling factor.
    - vp (sp.Matrix): Perspective vector. Can be of shape (1, 3) or (3, 1).

    Returns:
    - sp.Matrix: Homogenous transformation matrix. Has shape (4, 4).
    """
    e = e.reshape(3, 1)
    vt = vt.reshape(3, 1)
    vp = vp.reshape(1, 3)

    mat_homo = mat_rot_gibbs(e, phi)
    mat_homo = mat_homo.row_join(vt)
    bottom_row = vp.row_join(sp.Matrix([[sf]]))
    mat_homo = mat_homo.col_join(bottom_row)

    return mat_homo


def mat_trans_homo_inv(
    e: sp.Matrix, phi: sp.Symbol, vt: sp.Matrix, sf: sp.Symbol, vp: sp.Matrix
) -> sp.Matrix:
    """
    Calculates the inverse homogenous transformation matrix given an axis of
    rotation, degree of rotation, vector of translation, scaling factor and
    vector of perspective.

    Parameters:
    - e (sp.Matrix): Axis of rotation. Can be of shape (1, 3) or (3, 1).
    - phi (sp.Symbol): Degrees of rotation. Has to be in degrees, not radians.
    - vt (sp.Matrix): Translation vector. Can be of shape (1, 3) or (3, 1).
    - sf (sp.Symbol): Scaling factor.
    - vp (sp.Matrix): Perspective vector. Can be of shape (1, 3) or (3, 1).

    Returns:
    - sp.Matrix: Inverse homogenous transformation matrix. Has shape (4, 4).
    """
    e = e.reshape(3, 1)
    vt = vt.reshape(3, 1)
    vp = vp.reshape(1, 3)

    mat_rot_inv = mat_rot_gibbs(e, phi).T
    mat_inv = mat_rot_inv
    mat_inv = mat_inv.row_join(-mat_rot_inv @ vt)
    bottom_row = vp.row_join(sp.Matrix([[sf]]))
    mat_inv = mat_inv.col_join(bottom_row)

    return mat_inv


def apply_homo_trans(mat: sp.Matrix, p: sp.Matrix) -> sp.Matrix:
    """
    Applies a homogenous transformation matrix to a vector.

    Parameters:
    - mat (sp.Matrix): Homogenous transformation matrix. Has to be of shape (4,
      4).
    - p (sp.Matrix): Vector to be transformed. Can be of shape (3, 1) or (1,
      3).

    Returns:
    - sp.Matrix: Vector after transformation. Has shape (4, 1).
    """
    p = p.reshape(3, 1)

    p = p.col_join(sp.Matrix([[1]]))
    return mat @ p
