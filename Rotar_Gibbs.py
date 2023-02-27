import numpy as np
from numpy.linalg import norm
from math import sin, cos, pi

# Vector que se desea rotar
u = np.array([10, 2, -5])
# Ángulo en el que se va rotar el vector (en grados)
ang = 30
# Eje sobre el que se va rotar (no tiene que ser unitario)
eje = np.array([1 / np.sqrt(13), 1 / np.sqrt(13), -1 / np.sqrt(13)])

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

W = Q @ u

print(W)
