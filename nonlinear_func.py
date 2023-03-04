import numpy as np

# sin(x1(x1 + x2)) cos(x3 + x4x5) sin(ex5 + ex6 − x2).

def func1(xs):
    return np.sin(xs[:, 1]* (xs[:, 1])) * np.cos(xs[:, 3]+ xs[:, 4] * xs[:, 5]) * np.sin(np.exp(xs[:, 5])+np.exp(xs[:, 6])-xs[:, 2])


# sin(x3(x1 + x2)) cos(x3 + x4x5) sin(ex5 + ex6 − x2).s

def func2(xs):
    return np.sin(xs[:, 2]* (xs[:, 3])) * np.cos(xs[:, 5]+ xs[:, 1] * xs[:, 4]) * np.sin(2 * np.exp(xs[:, 2]) - np.exp(xs[:, 1])-xs[:, 3])
