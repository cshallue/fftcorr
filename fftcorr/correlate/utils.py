import numpy as np
import scipy.special


def triplefact(j1, j2, j3):
    """Returns g!/((g - j1)!(g - j2)!(g - j3!)), where g = (j1 + j2 + j3)/2."""
    jhalf = (j1 + j2 + j3) / 2
    return (scipy.special.factorial(jhalf) /
            scipy.special.factorial(jhalf - j1) /
            scipy.special.factorial(jhalf - j2) /
            scipy.special.factorial(jhalf - j3))


def threej(j1, j2, j3):
    """Returns the {j1,j2,j3; 0,0,0} Wigner 3j symbol.

    See https://mathworld.wolfram.com/Wigner3j-Symbol.html.
    """
    j = j1 + j2 + j3
    if j % 2:
        return 0  # Must be even
    if (j1 + j2 < j3) or (j2 + j3 < j1) or (j3 + j1 < j2):
        return 0  # Check legel triangle
    return (-1)**(j / 2) * triplefact(
        j1, j2, j3) / (triplefact(2 * j1, 2 * j2, 2 * j3) * (j + 1))**0.5
    # DJE did check this against Wolfram


def Mkl_calc(k, ell, flist):
    """Computes the matrix element from SE15, eq. 9."""
    # This is the matrix element from SE15, eq 9
    jlist = 2 * np.arange(len(flist))  # The f list is j=0,2,4,...
    return (2 * k + 1) * np.sum(
        [threej(ell, j, k)**2 * fj for j, fj in zip(jlist[1:], flist[1:])])


def boundary_correct(xi_raw, fRR):
    """Performs the boundary correction from SE15, eq. 10."""
    # In eq. 10, N[k] = 0 for odd k (by eq. 5, since NN is even in mu)
    # and A[k][l] = 0 when k+l is odd (by eq. 8, since f[j] = 0 for odd j and
    # the Wigner symbol is zero for l+j+k odd). Thus, we can replace eq. 10 with
    # a matrix equation of the same form but where all odd indexes of N are
    # removed and all odd rows and columns of A are removed.
    num_ell, num_s = fRR.shape
    xi = np.zeros_like(xi_raw)
    for s in range(num_s):
        A = np.identity(num_ell)
        for k in range(num_ell):
            for l in range(num_ell):
                # Remember that the k's and ell's are indexed 0,2,4...
                A[k][l] += Mkl_calc(2 * k, 2 * l, fRR[:, s])
        xi[:, s] = np.linalg.solve(A, xi_raw[:, s])
    return xi


def compute_xi(hist_corrNN, hist_corrRR):
    """Computes the anisotropic 2PCF from multipole moments of NN and RR.

    NN(s, mu) and RR(s, mu) are as defined in SE15, eq. 5.

    Args:
      hist_corrNN: Array [num_ell, num_s] of multipole moments of NN(s, mu)
      hist_corrRR: Array [num_ell, num_s] of multipole moments of RR(s, mu)
    """
    fRR = hist_corrRR / hist_corrRR[0]  # f_j from SE15 eq. 8
    xi_raw = hist_corrNN / hist_corrRR[0]  # LHS of SE15 eq. 10
    xi = boundary_correct(xi_raw, fRR)  # Solve SE15 eq. 10
    return xi