import numpy as np
from numpy.linalg import norm
from math import sin, cos, pi


def rotate(u: np.ndarray, ang: float, eje: np.ndarray) -> np.ndarray:
    """
    Rotate the vector 'u' around the axis 'eje' 'ang' degrees.

    Parameters:
    u (np.ndarray): Vector to be rotated.
    ang (float): Angle, in degrees, used to rotate the vector 'u'.
    eje (np.ndarray): Axis around which to rotate the vector 'u'.

    Returns:
    np.ndarray: Rotated vector 'u' around the axis 'eje' with a rotation of 'ang' degrees.
    """

    ang = ang * (pi / 180)
    uni = eje / norm(eje)
    Se = np.array(
        [
            np.array([0, -uni[2], uni[1]]),
            np.array([uni[2], 0, -uni[0]]),
            np.array([-uni[1], uni[0], 0]),
        ]
    )
    Q = (
        np.outer(uni, uni)
        + (np.identity(3) - np.outer(uni, uni)) * cos(ang)
        + Se * sin(ang)
    )
    return Q @ u


def main():
    # Vector que se desea rotar
    u = np.array([10, 2, -5])
    # Ángulo en el que se va rotar el vector (en grados)
    ang = 30
    # Eje sobre el que se va rotar (no tiene que ser unitario)
    eje = np.array([1 / np.sqrt(13), 1 / np.sqrt(13), -1 / np.sqrt(13)])

    W = rotate(u, ang, eje)

    print(W)


if __name__ == "__main__":
    main()
