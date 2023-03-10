import numpy as np


def rotate(e: np.ndarray, theta: float) -> np.ndarray:
    _theta = np.deg2rad(theta)

    ex: float = e[0, 0]
    ey: float = e[1, 0]
    ez: float = e[2, 0]

    se = np.array(
        [np.array([0, -ez, ey]), np.array([ez, 0, -ex]), np.array([-ey, ex, 0])]
    )

    return e @ e.T + (np.eye(3) - e @ e.T) * np.cos(_theta) + se * np.sin(_theta)
