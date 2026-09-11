import numpy as np
from matplotlib import pyplot as plt



x = np.linspace(0, 1, 100)
conditions = [(0 <= x) & (x <= 0.5), (0.5 < x) & (x <= 1)]

s1_x = np.piecewise(
    x,
    conditions,
    [lambda x: -1, lambda x: -3]
)
s2_x = np.piecewise(
    x,
    conditions,
    [lambda x: 2, lambda x: 0]
)
s3_x = np.piecewise(
    x,
    conditions,
    [lambda x: 1, lambda x: -3]
)

f1_x = np.piecewise(
    x,
    conditions,
    [lambda x: 1, lambda x: -1]
)
f2_x = np.piecewise(
    x,
    conditions,
    [lambda x: 1, lambda x: 1]
)

funcs = (f1_x, f2_x)
e = np.zeros((len(funcs), len(funcs)))
for i in range(len(funcs)):
    for j in range(len(funcs)):
        e[i, j] = np.trapezoid(funcs[i] * funcs[j], x)

print(e)
