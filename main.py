import sympy as sp

sp.init_printing()

ex, ey, ez = sp.symbols("ex ey ez")

e = sp.Matrix([[ex], [ey], [ez]])

uni = e / e.norm()

sp.pprint(uni)
