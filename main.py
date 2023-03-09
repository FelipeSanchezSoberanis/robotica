from rotar_gibbs import rotate
import numpy as np
import sympy as sp


def main():
    ex, ey, ez = sp.symbols("ex ey ez")
    theta = 30
    u = np.array([1, 1, 1])
    eje = np.array([ex, ey, ez])

    W = rotate(u, theta, eje)

    print(W)


if __name__ == "__main__":
    main()
