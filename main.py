import operations.numeric as num_op
import operations.symbolic as sym_op
import numpy as np
import sympy as sp


def translate(x: sp.Symbol, y: sp.Symbol, z: sp.Symbol):
    part1 = np.append(np.eye(3), sp.Matrix([[x], [y], [z]]), axis=1)
    result = np.append(part1, np.array([[0, 0, 0, 1]]), axis=0)
    print(result)


def main():
    #  T = np.array([[0.86, -0.5, 0, 10], [0.5, 0.86, 0, 5], [0, 0, 1, 0], [0, 0, 0, 1]])
    #  P1 = np.array([[3], [7], [0], [1]])
    #  P2 = T.dot(P1)
    x, y, z = sp.symbols("x y z")
    translate(x, y, z)


if __name__ == "__main__":
    main()
