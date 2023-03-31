import numpy as np


def mat_rot_gibbs(e: np.ndarray, phi: float) -> np.ndarray:
    """
    Calculates the Gibbs rotation matrix when rotating "phi" degrees around the
    axis "e".

    Parameters:
    - e (np.ndarray): Axis of rotation. Can be of shape (1, 3) or (3, 1).
    - phi (float): Degrees of rotation. Has to be in degrees, not radians.

    Returns:
    - np.ndarray: Gibbs rotation matrix. Has shape (3, 3).
    """
    e = e.reshape(3, 1)
    phi = np.deg2rad(phi)

    ex, ey, ez = e[0, 0], e[1, 0], e[2, 0]
    se = np.array([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]], dtype=float)
    mat_rot = e @ e.T + np.cos(phi) * (np.eye(3) - e @ e.T) + np.sin(phi) * se

    return mat_rot


def mat_trans_homo(
    e: np.ndarray, phi: float, vt: np.ndarray, sf: float, vp: np.ndarray
) -> np.ndarray:
    """
    Calculates the homogenous transformation matrix given an axis of rotation,
    degree of rotation, vector of translation, scaling factor and vector of
    perspective.

    Parameters:
    - e (np.ndarray): Axis of rotation. Can be of shape (1, 3) or (3, 1).
    - phi (float): Degrees of rotation. Has to be in degrees, not radians.
    - vt (np.ndarray): Translation vector. Can be of shape (1, 3) or (3, 1).
    - sf (float): Scaling factor.
    - vp (np.ndarray): Perspective vector. Can be of shape (1, 3) or (3, 1).

    Returns:
    - np.ndarray: Homogenous transformation matrix. Has shape (4, 4).
    """
    e = e.reshape(3, 1)
    vt = vt.reshape(3, 1)
    vp = vp.reshape(1, 3)

    mat_homo = mat_rot_gibbs(e, phi)
    mat_homo = np.append(mat_homo, vt, axis=1)
    bottom_row = np.append(vp, np.array([[sf]]), axis=1)
    mat_homo = np.append(mat_homo, bottom_row, axis=0)

    return mat_homo
