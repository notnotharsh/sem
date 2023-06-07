import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
from scipy.interpolate import CubicSpline

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, linewidth=100000)

def interp_mat(xo, xi):
    """
    Compute the interpolation matrix from xi to xo
    """
    no = xo.shape[0]
    ni = xi.shape[0]
    a = np.ones(ni)  # Initialize a as an array of ones with length ni

    for i in range(ni):
        for j in range(i):
            a[i] *= (xi[i] - xi[j])
        for j in range(i + 1, ni):
            a[i] *= (xi[i] - xi[j])
    
    a = 1 / a  # Compute the reciprocal of each element in a using element-wise division
    J = np.zeros((no, ni))  # Initialize J as a 2D array of zeros with dimensions (no, ni)
    s = np.ones(ni)  # Initialize s as an array of ones with length ni
    t = np.ones(ni)  # Initialize t as an array of ones with length ni

    for i in range(no):
        x = xo[i]
        for j in range(1, ni):
            s[j] = s[j - 1] * (x - xi[j - 1])
            t[ni - 1 - j] = t[ni - j] * (x - xi[ni - j])
        J[i] = a * s * t
    return J

def deriv_mat(x):
    """
    Compute the derivative matrix from x to x
    """
    ni = x.shape[0]
    a = np.ones(ni)  # Initialize a as an array of ones with length ni

    for i in range(ni):
        for j in range(i):
            a[i] *= (x[i] - x[j])
        for j in range(i + 1, ni):
            a[i] *= (x[i] - x[j])

    a = 1 / a  # Compute the reciprocal of each element in a using element-wise division

    d = np.zeros((ni, ni))  # Initialize d as a 2D array of zeros with dimensions (ni, ni)

    for j in range(ni):
        for i in range(ni):
            d[i, j] = x[i] - x[j]
        d[j, j] = 1
    
    d = 1 / d  # Compute the reciprocal of each element in d using element-wise division

    for i in range(ni):
        d[i, i] = 0
        d[i, i] = np.sum(d[i, :])

    for j in range(ni):
        for i in range(ni):
            if i != j:
                d[i, j] = a[j] / (a[i] * (x[i] - x[j]))

    return d

def zwgll(p):
    """
    Computes the p+1 Gauss-Lobatto-Legendre nodes z on [-1,1]
    and the p+1 weights w.
    """
    n = p + 1

    z = np.zeros(n)
    w = np.zeros(n)

    z[0] = -1
    z[n - 1] = 1

    if p > 1:
        if p == 2:
            z[1] = 0
        else:
            M = np.zeros((p - 1, p - 1))
            for i in range(p - 2):
                M[i, i + 1] = (1 / 2) * np.sqrt(((i + 1) * (i + 3)) / ((i + 3 / 2) * (i + 5 / 2)))
                M[i + 1, i] = M[i, i + 1]
            D, V = la.eigh(M)
            z[1:p] = np.sort(D)

    w[0] = 2 / (p * n)
    w[n - 1] = w[0]

    for i in range(1, p):
        x = z[i]
        z0 = 1
        z1 = x

        for j in range(p - 1):
            z2 = x * z1 * (2 * j + 3) / (j + 2) - z0 * (j + 1) / (j + 2)
            z0 = z1
            z1 = z2

        w[i] = 2 / (p * n * z2 * z2)

    return z, w

def semhat(N):
    z, w = zwgll(N)
    Bh = np.diag(w)
    Dh = deriv_mat(z)

    Ch = Bh@Dh

    Ah = Dh.T@Bh@Dh
    Ah = Ah / 2 + Ah.T / 2 # symmetry

    for i in range(N):
        Ah[i, i] -= np.sum(Ah[i, :]) # null space
    
    return Ah, Bh, Ch, Dh, z, w

def poisson_sem_iso(bdy_points):

    # nonintegers
    k = .01
    l = .89
    
    # degrees
    N0 = 4
    N1 = 6
    Ns = 2

    N_arr = []
    err_arr = []

    N = N0

    Ah, Bh, Ch, Dh, z, w = semhat(N)
    Ao, Bo, Co, Do, zo, wo = semhat(N + 2)

    Jh = interp_mat(zo, z)
    Bf = Jh.T@Bo@Jh
    # Bh = Bf # if you want to use the full mass matrix

    # size of each dimension (original is 2)
    Lx = 2
    Ly = 2

    x = Lx / 2 * (z) # x is GLL nodes in 0, 1
    y = Ly / 2 * (z) # y is GLL nodes in 0, 1

    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # restriction matrix (changes on boundary conditions)
    Rx = np.eye(N + 1)
    Rx = Rx[1:N, :]

    Ry = np.eye(N + 1)
    Ry = Ry[1:N, :]

    Ax = (2 / Lx) * Rx@Ah@Rx.T
    Ax = (Ax + Ax.T) / 2 # symmetry
    
    Bbx = (Lx / 2) * Bh
    Bx = Rx@Bbx@Rx.T

    Ay = (2 / Ly) * Ry@Ah@Ry.T
    Ay = (Ay + Ay.T) / 2 # symmetry
    
    Bby = (Ly / 2) * Bh
    By = Ry@Bby@Ry.T

    Abx = (Ah + Ah.T) / Lx
    Aby = (Ah + Ah.T) / Ly

    n = Ax.shape[0]
    n_bar = Abx.shape[0]

    Dx, Sx = la.eigh(Ax, Bx)
    Dx = sparse.csr_matrix(np.diag(Dx))

    Dy, Sy = la.eigh(Ay, By)
    Dy = sparse.csr_matrix(np.diag(Dy))

    Dx_bar, Sx_bar = la.eigh(Abx, Bbx)
    Dy_bar, Sy_bar = la.eigh(Aby, Bby)
    Dx_bar = sparse.csr_matrix(np.diag(Dx_bar))
    Dy_bar = sparse.csr_matrix(np.diag(Dy_bar))

    for j in range(n): # normalize eigenvectors
        Sx[:,j] /= np.sqrt(Sx[:,j].T@Bx@Sx[:,j])
        Sy[:,j] /= np.sqrt(Sy[:,j].T@By@Sy[:,j])
    
    for j in range(n_bar):
        Sx_bar[:,j] /= np.sqrt(Sx_bar[:,j].T@Bbx@Sx_bar[:,j])
        Sy_bar[:,j] /= np.sqrt(Sy_bar[:,j].T@Bby@Sy_bar[:,j])
    
    Ix = sparse.eye(n)
    Iy = sparse.eye(n)

    Ibx = sparse.eye(n_bar)
    Iby = sparse.eye(n_bar)

    D = sparse.kron(Iy, Dx) + sparse.kron(Dy, Ix)
    D = D.diagonal()
    D = np.reshape(D, (n, n))

    Db = sparse.kron(Iby, Dx_bar) + sparse.kron(Dy_bar, Ibx)
    Db = Db.diagonal()
    Db = np.reshape(Db, (n_bar, n_bar))

    Sxi_bar = np.linalg.inv(Sx_bar)
    Syi_bar = np.linalg.inv(Sy_bar)

    XsYs = []
    f = 0 * X
    ue = bdy_points
    ub = np.zeros_like(ue)

    ub[0, :] = ue[0, :]
    ub[:, 0] = ue[:, 0]
    ub[-1, :] = ue[-1, :]
    ub[:, -1] = ue[:, -1]

    Bf = Bbx@f@Bby.T
    Bf = Rx@Bf@Ry.T

    inhom_effect = Sxi_bar.T @ np.multiply((Sxi_bar@ub@Syi_bar.T), Db) @ Syi_bar
    inhom_effect = Rx@inhom_effect@Ry.T

    u = Sx @ np.divide((Sx.T @ (Bf - inhom_effect) @ Sy), D) @ Sy.T
    ub += Rx.T@u@Ry

    return ub

def element_meshing(x_points, y_points):
    X = poisson_sem_iso(x_points)
    Y = poisson_sem_iso(y_points)

    uf = np.linspace(-1, 1, 30)
    z, w = zwgll(4)
    J = interp_mat(uf, z)

    Xf = J@X@J.T
    Yf = J@Y@J.T

    fig = plt.figure(figsize=plt.figaspect(1))
    plt.axis('off')
    plt.title(r"interpolated element mesh")
    
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_wireframe(Xf, Yf, 0 * Xf, linewidth=2, alpha=1, color='r')
    ax.plot_wireframe(X, Y, 0 * X, linewidth=2, alpha=1, color='b')

    plt.show()


x_points = np.array([[.6, 1, 1.8, 2.7, 3.2], [.7, 0, 0, 0, 3.25], [0.8, 0, 0, 0, 3.3], [.8, 0, 0, 0, 3.7], [.5, .7, 1.9, 3.3, 4.0]])
y_points = np.array([[1.7, 1.8, 1.8, 1.7, 1.6], [2.1, 0, 0, 0, 2.0], [3.1, 0, 0, 0, 3.2], [4.05, 0, 0, 0, 4.3], [4.4, 4.6, 5.2, 5.1, 5.0]])

element_meshing(x_points, y_points)
