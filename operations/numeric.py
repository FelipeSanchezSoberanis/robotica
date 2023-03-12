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


def trotar(e: np.ndarray, theta: float) -> np.ndarray:
    _theta: float = np.deg2rad(theta)

    ex: float = e[0, 0]
    ey: float = e[0, 1]
    ez: float = e[0, 2]

    se = np.array(
        [np.array([0, -ez, ey]), np.array([ez, 0, -ex]), np.array([-ey, ex, 0])]
    )

    parte1: np.ndarray = (
        e.T @ e + np.cos(_theta) * (np.eye(3) - e.T @ e) + np.sin(_theta) * se
    )
    parte2 = np.append(parte1, np.array([[0], [0], [0]]), axis=1)
    result = np.append(parte2, np.array([[0, 0, 0, 1]]), axis=0)

    return result
