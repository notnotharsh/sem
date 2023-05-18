import numpy as np
import numpy.linalg as la

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