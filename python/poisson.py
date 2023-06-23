import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

from scipy.interpolate import CubicSpline
import scipy.integrate as integrate

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

def poisson_sem_iso(bdy_points, N, func=None):

    # nonintegers
    # k = .01
    # l = .89
     
    # degrees
    # N0 = 4
    # N1 = 6
    # Ns = 2

    N_arr = []
    err_arr = []

    # N = N0

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
    if func is None:
        f = 0 * X
    else:
        f = func(X, Y)
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

def element_meshing(x_points, y_points, N):

    n = x_points.shape[0]

    old_X = poisson_sem_iso(x_points, n - 1)
    old_Y = poisson_sem_iso(y_points, n - 1)

    x_points[0, :], y_points[0, :] = gll_redistribution(x_points[0, :], y_points[0, :])
    x_points[-1, :], y_points[-1, :] = gll_redistribution(x_points[-1, :], y_points[-1, :])
    x_points[:, 0], y_points[:, 0] = gll_redistribution(x_points[:, 0], y_points[:, 0])
    x_points[:, -1], y_points[:, -1] = gll_redistribution(x_points[:, -1], y_points[:, -1])

    X = poisson_sem_iso(x_points, n - 1)
    Y = poisson_sem_iso(y_points, n - 1)

    uf = zwgll(N - 1)[0]
    z, w = zwgll(4)
    J = interp_mat(uf, z)

    Xf = J@X@J.T
    Yf = J@Y@J.T

    """

    old_Xf = J@old_X@J.T
    old_Yf = J@old_Y@J.T

    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.axis('off')
    plt.title(r"interpolated element mesh, without and with GLL adjustments")

    r, s = np.meshgrid(zwgll(N - 1)[0], zwgll(N - 1)[0], indexing="ij")

    
    ax = fig.add_subplot(1, 2, 1)
    
    ax.plot(old_Xf, old_Yf, linewidth=1, alpha=1, color='r')
    ax.plot(old_Xf.T, old_Yf.T, linewidth=1, alpha=1, color='r')
    
    ax.plot(old_X, old_Y, linewidth=2, alpha=1, color='b')
    ax.plot(old_X.T, old_Y.T, linewidth=2, alpha=1, color='b')

    plt.gca().set_aspect('equal')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(Xf, Yf, linewidth=1, alpha=1, color='r')
    ax.plot(Xf.T, Yf.T, linewidth=1, alpha=1, color='r')
    
    ax.plot(X, Y, linewidth=2, alpha=1, color='b')
    ax.plot(X.T, Y.T, linewidth=2, alpha=1, color='b')

    plt.gca().set_aspect('equal')
    plt.show() 
    """

    return Xf, Yf

def gll_redistribution(x_top, y_top):
    n = x_top.shape[0]
    t_top = np.zeros_like(x_top)
    point_sum = 0

    for i in range(n - 1):
        point_sum += np.hypot(x_top[i + 1] - x_top[i], y_top[i + 1] - y_top[i])
        t_top[i + 1] = point_sum

    t_top /= (point_sum / 2)
    t_top -= 1

    csx = CubicSpline(t_top, x_top)
    csy = CubicSpline(t_top, y_top)

    s_top = np.zeros_like(t_top)
    point_sum = 0

    for i in range(n - 1):
        point_sum += integrate.quad(lambda t: np.hypot(csx(t, 1), csy(t, 1)), t_top[i], t_top[i + 1])[0]
        s_top[i + 1] = point_sum

    s_top /= (point_sum / 2)
    s_top -= 1

    ccsx = CubicSpline(s_top, x_top)
    ccsy = CubicSpline(s_top, y_top)

    s_gll = zwgll(n - 1)[0]
    s_many = np.linspace(-1, 1, 1000)

    x_gll = ccsx(s_gll)
    y_gll = ccsy(s_gll)

    return x_gll, y_gll

def uxy(x, y):
    return np.sin(x) * np.exp(y) / (2)

def fxy(x, y):
    return 0

x_points = np.array([[.6, 1, 1.8, 2.7, 3.2], [.7, 0, 0, 0, 3.25], [0.8, 0, 0, 0, 3.3], [.8, 0, 0, 0, 3.7], [.5, .7, 1.9, 3.3, 4.0]])
y_points = np.array([[1.7, 1.8, 1.8, 1.7, 1.6], [2.1, 0, 0, 0, 2.0], [3.1, 0, 0, 0, 3.2], [4.05, 0, 0, 0, 4.3], [4.4, 4.6, 5.2, 5.1, 5.0]])

# x_points = np.array([[0, .25, .5, .75, 1], [0, .25, .5, .75, 1], [0, .25, .5, .75, 1], [0, .25, .5, .75, 1], [0, .25, .5, .75, 1]]) * 2 - 1
# y_points = x_points.T * 3
# x_points *= 3


N0 = 6
N1 = 40
Ns = 2

N_arr = []
err_arr = []

for N in range(N0, N1, Ns):

    print(N)

    Ah, Bh, Ch, Dh, z, w = semhat(N)
    X, Y = np.meshgrid(zwgll(N)[0], zwgll(N)[0], indexing='ij')
    Xf, Yf = element_meshing(x_points, y_points, N + 1)


    ue = uxy(Xf, Yf)
    f = fxy(Xf, Yf)

    ub = np.zeros_like(ue)
    ub[0, :] = ue[0, :]
    ub[:, 0] = ue[:, 0]
    ub[-1, :] = ue[-1, :]
    ub[:, -1] = ue[:, -1]

    Ih = sparse.eye(N + 1)

    xr = Dh@Xf
    xs = Xf@Dh.T
    yr = Dh@Yf
    ys = Yf@Dh.T

    J = xr * ys - xs * yr

    rx = ys / J
    ry = -xs / J
    sx = -yr / J
    sy = xr / J 

    Dr = sparse.kron(Ih, Dh)
    Ds = sparse.kron(Dh, Ih)

    Rdx = sparse.csr_matrix(np.diag(np.concatenate(rx.T)))
    Sdx = sparse.csr_matrix(np.diag(np.concatenate(sx.T)))
    Rdy = sparse.csr_matrix(np.diag(np.concatenate(ry.T)))
    Sdy = sparse.csr_matrix(np.diag(np.concatenate(sy.T)))

    Dex = Rdx.T@Dr + Sdx.T@Ds
    Dey = Rdy.T@Dr + Sdy.T@Ds

    B = sparse.csr_matrix(np.diag(np.concatenate(J.T)))@sparse.kron(Bh, Bh)

    Rx = np.eye(N + 1)
    Rx = Rx[1:N, :]

    Ry = np.eye(N + 1)
    Ry = Ry[1:N, :]

    R = np.kron(Ry, Rx)

    """ without diagonalization"""

    ubv = np.concatenate(ub.T)

    Ab = Dex.T@B@Dex + Dey.T@B@Dey
    A = R@Ab@R.T

    rhs = -R@Ab@ubv
    ub += np.reshape(R.T@np.linalg.solve(A, rhs), (N + 1, N + 1)).T

    cosex = np.cos(Xf)
    sinx = np.sin(Xf)
    sinxv = sparse.csr_matrix(np.concatenate(sinx.T)).T

    print(Dex.shape, sinxv.shape)

    cosxv = Dex@sinxv

    print(cosxv.shape)
    cosx = np.reshape(cosxv, (N + 1, N + 1)).T.A
    
    er = ub - ue
    err = np.linalg.norm(er, np.inf)

    N_arr.append(N)
    err_arr.append(err)

    if (N >= N1 - Ns):
        fig = plt.figure(figsize=plt.figaspect(0.33))
        plt.axis('off')
        plt.title(r"spectral element solution (sem, theo, error) to $\nabla^2 u = 0$, deformed mesh, inhomogenous dirichlet boundary conditions")

        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.plot_wireframe(Xf, Yf, ub, linewidth=0.5, alpha=0.5)

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot_wireframe(Xf, Yf, ue, linewidth=0.5, alpha=0.5)
        ax.plot_wireframe(Xf, Yf, ub, linewidth=0.5, alpha=0.5, color='r')

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.plot_wireframe(Xf, Yf, er, linewidth=0.5, alpha=0.5)
        ax.set_zlim3d(-.01, .01)

        plt.show()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  
plt.title("SEM 2D Poisson solutions - error vs. number of nodes")
ax.plot(N_arr, err_arr)
ax.set_yscale('log')
plt.show()
