import operations.numeric as num_op
import operations.symbolic as sym_op
import numpy as np
import sympy as sp


def main():
    theta = sp.symbols("theta")
    sp.pprint(sym_op.trotar(sp.Matrix([[1, 0, 0]]), theta))


if __name__ == "__main__":
    main()
