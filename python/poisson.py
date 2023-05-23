import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

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

def poisson_sem():

    # nonintegers
    k = 5.3
    l = 6.2
    
    # degrees
    N0 = 2
    N1 = 60
    Ns = 2

    N_arr = []
    err_arr = []

    for N in range(N0, N1, Ns):

        Ah, Bh, Ch, Dh, z, w = semhat(N)
        Ao, Bo, Co, Do, zo, wo = semhat(N + 2)

        Jh = interp_mat(zo, z)
        Bf = Jh.T@Bo@Jh
        # Bh = Bf # if you want to use the full mass matrix

        # size of each dimension (original is 2)
        Lx = 1
        Ly = 1

        x = Lx / 2 * (z + np.ones(z.shape[0])) # x is GLL nodes in 0, 1
        y = Ly / 2 * (z + np.ones(z.shape[0])) # y is GLL nodes in 0, 1

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

        f = np.sin(np.pi * k * X) * np.sin(np.pi * l * Y)
        ue = f / (np.pi**2 * (k**2 + l**2))
        ub = np.zeros_like(ue)
        ub[-1, :] = ue[-1, :]
        ub[:, -1] = ue[:, -1]

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

        Bf = Bbx@f@Bby.T
        Bf = Rx@Bf@Ry.T

        Sxi_bar = np.linalg.inv(Sx_bar)
        Syi_bar = np.linalg.inv(Sy_bar)

        inhom_effect = Sxi_bar.T @ np.multiply((Sxi_bar@ub@Syi_bar.T), Db) @ Syi_bar
        inhom_effect = Rx@inhom_effect@Ry.T

        u = Sx @ np.divide((Sx.T @ (Bf - inhom_effect) @ Sy), D) @ Sy.T
        ub += Rx.T@u@Ry

        er = ue - ub
        err = np.linalg.norm(er, ord=np.inf)

        N_arr.append(N)
        err_arr.append(err)

        if N == N1 - Ns:
            fig = plt.figure(figsize=plt.figaspect(0.33))
            plt.axis('off')
            plt.title(r"spectral element solution to $\nabla^2 u = (\sin{\pi kx})(\sin{\pi ly})$, inhomogenous dirichlet boundary conditions")

            ax = fig.add_subplot(1, 3, 1, projection='3d')
            ax.plot_wireframe(X, Y, ub, linewidth=0.5, alpha=0.5)

            ax = fig.add_subplot(1, 3, 2, projection='3d')
            ax.plot_wireframe(X, Y, ue, linewidth=0.5, alpha=0.5)

            ax = fig.add_subplot(1, 3, 3, projection='3d')
            ax.plot_wireframe(X, Y, er, linewidth=0.5, alpha=0.5)
            ax.set_zlim3d(-.001, .001)

            plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    plt.title("SEM 2D Poisson solutions - error vs. number of nodes")
    ax.plot(N_arr, err_arr)
    ax.set_yscale('log')
    plt.show()

poisson_sem()