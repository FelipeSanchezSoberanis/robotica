import operations.numeric as num_op
import operations.symbolic as sym_op
import numpy as np
import sympy as sp


def main():
    ex, ey, ez = sp.symbols("ex ey ez")
    theta = sp.symbols("theta")

    print(num_op.rotate(np.array([[15], [2], [2]]), 30))
    sp.pprint(sym_op.rotate(sp.Matrix([[ex], [ey], [ez]]), theta))


if __name__ == "__main__":
    main()
