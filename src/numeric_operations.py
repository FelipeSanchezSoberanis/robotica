import numpy as np
from enum import Enum
import numpy.typing as npt


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


def mat_trans_homo_inv(
    e: np.ndarray, phi: float, vt: np.ndarray, sf: float, vp: np.ndarray
) -> np.ndarray:
    """
    Calculates the inverse homogenous transformation matrix given an axis of
    rotation, degree of rotation, vector of translation, scaling factor and
    vector of perspective.

    Parameters:
    - e (np.ndarray): Axis of rotation. Can be of shape (1, 3) or (3, 1).
    - phi (float): Degrees of rotation. Has to be in degrees, not radians.
    - vt (np.ndarray): Translation vector. Can be of shape (1, 3) or (3, 1).
    - sf (float): Scaling factor.
    - vp (np.ndarray): Perspective vector. Can be of shape (1, 3) or (3, 1).

    Returns:
    - np.ndarray: Inverse homogenous transformation matrix. Has shape (4, 4).
    """
    e = e.reshape(3, 1)
    vt = vt.reshape(3, 1)
    vp = vp.reshape(1, 3)

    mat_rot_inv = mat_rot_gibbs(e, phi).T
    mat_inv = mat_rot_inv
    mat_inv = np.append(mat_inv, -mat_rot_inv @ vt, axis=1)
    bottom_row = np.append(vp, np.array([[sf]]), axis=1)
    mat_inv = np.append(mat_inv, bottom_row, axis=0)

    return mat_inv


def apply_homo_trans(mat: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Applies a homogenous transformation matrix to a vector.

    Parameters:
    - mat (np.ndarray): Homogenous transformation matrix. Has to be of shape
      (4, 4).
    - p (np.ndarray): Vector to be transformed. Can be of shape (3, 1) or (1,
      3).

    Returns:
    - np.ndarray: Vector after transformation. Has shape (4, 1).
    """
    p = p.reshape(3, 1)

    p = np.append(p, np.array([[1]]), axis=0)
    return mat @ p


class AngleMode(Enum):
    DEG = 1
    RAD = 2


class DHParameters:
    list_a: list[float]
    list_r: list[float]
    list_alpha: list[float]
    list_theta: list[float]
    angle_mode: AngleMode

    def __init__(self, angle_mode: AngleMode):
        self.angle_mode = angle_mode
        self.list_a = []
        self.list_r = []
        self.list_alpha = []
        self.list_theta = []

    def add_parameters(self, a: float, r: float, alpha: float, theta: float) -> None:
        self.list_a.append(a)
        self.list_r.append(r)
        self.list_alpha.append(alpha)
        self.list_theta.append(theta)

    def get_transformation_matrices(self) -> list[npt.NDArray]:
        transformation_matrices: list[npt.NDArray] = []

        for a, r, alpha, theta in zip(self.list_a, self.list_r, self.list_alpha, self.list_theta):
            if self.angle_mode == AngleMode.DEG:
                alpha = np.deg2rad(alpha)
                theta = np.deg2rad(theta)

            transformation_matrices.append(
                np.array(
                    [
                        [
                            np.cos(theta),
                            -np.sin(theta) * np.cos(alpha),
                            np.sin(theta) * np.sin(alpha),
                            a * np.cos(theta),
                        ],
                        [
                            np.sin(theta),
                            np.cos(theta) * np.cos(alpha),
                            -np.cos(theta) * np.sin(alpha),
                            a * np.sin(theta),
                        ],
                        [0, np.sin(alpha), np.cos(alpha), r],
                        [0, 0, 0, 1],
                    ]
                )
            )

        return transformation_matrices

    def get_final_transformation_matrix(self) -> npt.NDArray:
        final_transformation_matrix = np.eye(4)

        for matrix in self.get_transformation_matrices():
            final_transformation_matrix = final_transformation_matrix @ matrix

        return final_transformation_matrix
