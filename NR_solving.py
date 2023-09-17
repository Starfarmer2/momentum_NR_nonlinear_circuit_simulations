import numpy as np
from sympy import *

INITIAL_GUESS = 10
Xn = INITIAL_GUESS

x, y, z = symbols("x y z")
f = x**2 + 1

#
# diffeq = Eq(f(x).diff(x, x) - 2*f(x).diff(x) + f(x), sin(x))
# f(x).diff(x)
plot(f,f.diff(x),show=True)