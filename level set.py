import matplotlib.pyplot as plt
import numpy as np

BETA = 2e-2
LAMBDA = 0.1
vt = 0.7


def f1(x, y, l):
    return BETA * ((x - vt) * y - 1 / 2 * pow(y, 2)) * (1 + l * y)


def f2(x, y, l):
    return 0.5 * BETA * pow(x - vt, 2) * (1 + l * y)


# def f4(x,y,l):
#     vt=0.7
#     ret = f1(x,y,l)
#     for i in range(len(y)):
#             if x[i]-vt>y[i]:
#                 ret[i] = f2(x[i],y[i],l)
#     return ret

def f5(x, y, l):
    if (x - vt > y):
        return f1(x, y, l)
    else:
        return f2(x, y, l)


def f5d1(x, y, l):
    if (x - vt > y):
        return BETA * y * (1 + l * y)
    else:
        return BETA * (x - vt) * (1 + l * y)


def f5d2(x, y, l):
    if (x - vt > y):
        return BETA * ((x - vt) + 2 * l * y * (x - vt) - y - 3 / 2 * l * pow(y, 2))
    else:
        return BETA / 2 * l * pow(x - vt, 2)


def f6(x, y, l):
    return (x - 9) / 1e2 + x / 2e2


def f6d1(x, y, l):
    return 1 / 1e2 + 1 / 2e2


def f6d2(x, y, l):
    return 0

X, Y = np.meshgrid(np.arange(-10, 20, 1,dtype='float64'), np.arange(-10, 20, 1,dtype='float64'))
Xp, Yp = np.meshgrid(np.arange(-10, 20, 0.5,dtype='float64'), np.arange(-10, 20, 0.5,dtype='float64'))

x_shape = X.shape
xp_shape = Xp.shape

F1 = np.zeros_like(X)
F1X = np.zeros_like(X)
F1Y = np.zeros_like(X)

F2 = np.zeros_like(X)
F2X = np.zeros_like(X)
F2Y = np.zeros_like(X)

F3X = np.zeros_like(X)
F3Y = np.zeros_like(X)

for i in range(len(X)):
    for j in range(len(X[i])):
        x = X[i][j]
        y = Y[i][j]
        F1[i][j] = f5(x, y, LAMBDA) + (y - 9) / 1e2
        F2[i, j] = f6(x, y, LAMBDA)

for i in range(x_shape[0]):
    for j in range(x_shape[1]):
        F1X[i, j] = f5d1(X[i, j], Y[i, j], LAMBDA)
        F1Y[i, j] = f5d2(X[i, j], Y[i, j], LAMBDA)
        F2X[i, j] = f6d1(X[i, j], Y[i, j], LAMBDA)
        F2Y[i, j] = f6d2(X[i, j], Y[i, j], LAMBDA)

        invmatrix = np.linalg.inv(np.array([[F1X[i, j], F1Y[i, j]],
                                            [F2X[i, j], F2Y[i, j]]]))
        F3X[i, j], F3Y[i, j] = -np.matmul(invmatrix, np.array([F1[i, j], F2[i, j]]))

F1p = np.zeros_like(Xp)
F1Xp = np.zeros_like(Xp)
F1Yp = np.zeros_like(Xp)

F2p = np.zeros_like(Xp)
F2Xp = np.zeros_like(Xp)
F2Yp = np.zeros_like(Xp)

F3Xp = np.zeros_like(Xp)
F3Yp = np.zeros_like(Xp)

for i in range(len(Xp)):
    for j in range(len(Xp[i])):
        x = Xp[i][j]
        y = Yp[i][j]
        F1p[i][j] = f5(x, y, LAMBDA) + (y - 9) / 1e2
        F2p[i, j] = f6(x, y, LAMBDA)

for i in range(xp_shape[0]):
    for j in range(xp_shape[1]):
        F1Xp[i, j] = f5d1(Xp[i, j], Yp[i, j], LAMBDA)
        F1Yp[i, j] = f5d2(Xp[i, j], Yp[i, j], LAMBDA)
        F2Xp[i, j] = f6d1(Xp[i, j], Yp[i, j], LAMBDA)
        F2Yp[i, j] = f6d2(Xp[i, j], Yp[i, j], LAMBDA)

        invmatrix = np.linalg.inv(np.array([[F1Xp[i, j], F1Yp[i, j]],
                                            [F2Xp[i, j], F2Yp[i, j]]]))
        F3Xp[i, j], F3Yp[i, j] = -np.matmul(invmatrix, np.array([F1p[i, j], F2p[i, j]]))

