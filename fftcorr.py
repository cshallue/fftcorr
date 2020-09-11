"""
fftcorr.py: Daniel Eisenstein, July 2016

This implements the anisotropic 2-point correlation function based on the
method of Slepian & Eisenstein (2015).

Memory usage: 

Aside from the D & R data themselves, the largest usage is in 4 FFT grids.
Two are sized to N^3, and two to the Fourier conjugate, which are N^2*(N//2+1).

We loop over m for each ell, so we only have to deal with one at a time.
In detail, we build the density and store it and its FFT.  Then we build
a Ylm (config), multiply by Dens in place, FFT it to the conjugate buffer, 
multiply by Dfft in place, then iFFT into the second config-space buffer.

In principle, this could be cut back to 3 grids, but dealing with one grid
that has to change size from N^3 to N^2*(N//2+1) is annoying for now.

We build the Ylm's one slab at a time, so memory usage here is small.
We store the 2-d output only for a sub-tensor of the separations, which
is much smaller.

Timing:

Right now, assigning the grid points is sub-dominant, but when that changes to 
cloud-in-cell, it may get worse.

The FFTs (two per ell,m) are currently the dominant part, but forming the Ylm
is not negligible.

Multi-threading:

This should all be highly multi-threadable.  Aside from the FFTs, most of the 
remaining work is in big element-wise multiplies or conjugations.

Missing optimizations:

Should one care, the random points are being gridded twice, rather than storing
and co-adding a density grid.  That said, the current code is minimizing memory
usage, since the grid is probably bigger than the list of points.

"""


import numpy as np
import astropy.io.fits
import wcdm
import struct
from timeit import default_timer as timer
from scipy import interpolate
import scipy.special
import subprocess
import shlex
import os

# Timing

times = {}
last_time = timer()


def lapsed_time(name):
    global last_time
    now = timer()
    times[name] = now-last_time
    last_time = now

##################  Global parameters ##############


NGRID = 256         # The grid size for the main FFTs
MAX_SEP = 200.0	    # The maximum separation for correlation computation
DSEP = 10.0         # The binning of the separations
MAX_ELL = 2         # How many multipoles we'll compute (ell must be even!)
QPERIODIC = 0	    # If ==1, use periodic boundary conditions

# The best choices for BOSS.
RA_ROTATE = {
    "North": -142.5,
    "South": 44,
}

# And the cosmological parameters
COSMOLOGY = {
    "omega": 0.317
}


##################  Catalog class #####################

class Catalog(object):
    def __init__(self, pos, w):
        self.pos = pos
        self.w = w


##################  Grid class #####################

class Grid(object):
    """
    Data in this class:
    ngrid: The grid size
    max_sep: The maximum separation we'll be computing
    posmin: The minimum (x,y,z) of the data, including a buffer
    posmax: The maximum (x,y,z) of the data, including a buffer
    boxsize: The resulting cubic box size
    cell_size: The size per cell
    origin: Where the origin is in grid coordinates
    xcell, ycell, zcell: The centers of the grid cells, relative to the origin
    max_sep_cell: The number of cells we'll extract on all sides of zero lag,
        chosen so that max_sep is safely included
    corr_cell: The centers of the separation cells, in this submatrix
    rcorr: The radii of each 3-d cell, in this submatrix
    """

    def __init__(self, ngrid, posmin, posmax, max_sep):
        self.ngrid = ngrid
        self.max_sep = max_sep

        # Find the bounding box, including the appropriate buffer
        # Returns the minimum corner of the box and the cubic box size
        pad = max_sep / 1.5
        posmin -= pad
        posmax += pad
        boxsize = np.max(posmax - posmin)
        cell_size = boxsize / ngrid
        self.posmin = posmin
        self.posmax = posmax
        self.boxsize = boxsize
        self.cell_size = cell_size

        print("Adopting boxsize %f, cell size %f for a %d^3 grid." %
              (boxsize, cell_size, ngrid))
        # Keep track of the origin on the grid
        origin = np.add(*self.pos2grid(np.zeros(3)))
        print("Origin is at grid location ", origin)
        self.origin = origin

        # Make the x,y,z ingredients for the Ylm's
        self.xcell = np.arange(ngrid) + 0.5 - origin[0]
        self.ycell = np.arange(ngrid) + 0.5 - origin[1]
        self.zcell = np.arange(ngrid) + 0.5 - origin[2]

        # Set up the correlation submatrix calculation
        max_sep_cell = np.ceil(max_sep / cell_size)
        print("Correlating to %f will extend to +-%d cells." %
              (max_sep, max_sep_cell))
        if (ngrid < 2 * max_sep_cell + 1):
            raise ValueError(
                "Grid size is too small relative to separation request")
        # This list will be used for the Ylm call
        corr_cell = np.arange(-max_sep_cell, max_sep_cell+1)
        # Make the radial grid
        corr_grid = np.meshgrid(
            corr_cell, corr_cell, corr_cell, indexing="ij")
        rcorr = np.sqrt(
            corr_grid[0]**2+corr_grid[1]**2+corr_grid[2]**2)*cell_size
        del corr_grid
        self.max_sep_cell = max_sep_cell
        self.corr_cell = corr_cell
        self.rcorr = rcorr

    def pos2grid(self, pos):
        # Convert positions to grid locations. Also return the residuals
        # Note that this does not check that the resulting grid is within ngrid
        return np.divmod(pos - self.posmin, self.cell_size)

# End class Grid


################ Functions ###############################

def coord2pos(ra, dec, rz, ra_rotate):
    # Convert angular positions to our Cartesian basis
    return np.array([
        rz*np.cos(dec*np.pi/180.0)*np.cos((ra+ra_rotate)*np.pi/180.0),
        rz*np.cos(dec*np.pi/180.0)*np.sin((ra+ra_rotate)*np.pi/180.0),
        rz*np.sin(dec*np.pi/180.0)
    ]).T


def read_data_file(filename, fmt, Nrandom, cosmology, ra_rotate, minz=0.43, maxz=0.70):
    # Read a data file.  Use Nrandom to specify the number of randoms to use.
    # Nrandom>0 also triggers treating the weightings as by the random file
    # Return the Cartesian positions and
    print("Reading from %s" % filename)
    if fmt == "boss":
        with astropy.io.fits.open(filename) as hdulist:
            data = hdulist[1].data  # pylint:disable=no-member
        if Nrandom > 0:
            data = data[0:Nrandom]
        data = data[np.where((data["z"] > minz) & (data["z"] < maxz))]
        w = np.float64(data["weight_fkp"])
        if not Nrandom:
            w *= data["weight_systot"] * \
                (data["weight_cp"]+data["weight_noz"]-1.0)
    elif fmt == "patchy":
        if Nrandom > 0:
            dtypes = [
                ("ra", float), ("dec", float), ("z", float), ("nbar", float),
                ("bias", float), ("veto", float), ("fiber", float)]
        else:
            dtypes = [
                ("ra", float), ("dec", float), ("z", float), ("mass", float),
                ("nbar", float), ("bias", float), ("veto", float), ("fiber", float)]
        data = np.loadtxt(filename, dtypes)
        if Nrandom > 0:
            data = data[0:Nrandom]
        data = data[np.where((data["z"] > minz) & (data["z"] < maxz))]
        w = data["veto"]*data["fiber"]/(1+1e4*data["nbar"])
    else:
        raise ValueError("Unrecognized fmt: %s" % fmt)

    # Convert (ra, dec, Z) to (x, y, z).
    redshifts = np.linspace(0.0, maxz+0.1, 1000)
    rz = interpolate.InterpolatedUnivariateSpline(
        redshifts, 2997.92*wcdm.coorddist(redshifts, cosmology["omega"], -1, 0))
    print("Done computing cosmological distances.")
    pos = coord2pos(data["ra"], data["dec"], rz(data["z"]), ra_rotate)

    print("Using %d galaxies, total weight %g" % (len(pos), np.sum(w)))
    print("Done reading and trimming data.")
    return Catalog(pos, w)


def makeYlm(ell, m, xcell, ycell, zcell):
    # We're not actually returning Ylm here.
    # m>0 will return Re(Y_lm)*sqrt(2)
    # m<0 will return Im(Y_l|m|)*sqrt(2)
    # m=0 will return Y_l0
    # These are not including common minus signs, since we're only
    # using them in matched squares.
    # Input xcell, ycell, zcell are the x,y,z centers of the bins
    isqpi = np.sqrt(1.0/np.pi)
    if (m != 0):
        isqpi *= np.sqrt(2.0)    # Do this up-front, so we don't forget
    tiny = 1e-20
    ngrid = len(xcell)
    # TODO: If we could align these arrays, it would be better!
    y, z = np.meshgrid(ycell, zcell, indexing="ij")
    y2 = y*y     # These get used for each x, so good to cache
    y3 = y2*y     # These get used for each x, so good to cache
    y4 = y3*y     # These get used for each x, so good to cache
    z2 = z*z
    z3 = z2*z
    z4 = z3*z
    # TODO: If we could align this array, it would be better!
    Ylm = np.empty((ngrid,) * 3)
    # print("Type of Ylm: ", np.result_type(Ylm)  # This claims to be float64.)
    for j in range(0, ngrid):
        x = xcell[j]     # A scalar, just to make the code more symmetric
        x2 = x*x
        if (ell == 0) & (m == 0):
            Ylm[j, :, :] = isqpi/2.0

        # Here's the ell=2 set
        elif (ell == 2) & (m == 2):
            Ylm[j, :, :] = isqpi*np.sqrt(15./32.)*(x2-y2) \
                / (x2+y2+z2+tiny)
        elif (ell == 2) & (m == 1):
            Ylm[j, :, :] = isqpi*np.sqrt(15./8.)*x*z \
                / (x2+y2+z2+tiny)
        elif (ell == 2) & (m == 0):
            Ylm[j, :, :] = isqpi*np.sqrt(5./16.)*(2*z2-x2-y2) \
                / (x2+y2+z2+tiny)
        elif (ell == 2) & (m == -1):
            Ylm[j, :, :] = isqpi*np.sqrt(15./8.)*y*z \
                / (x2+y2+z2+tiny)
        elif (ell == 2) & (m == -2):
            Ylm[j, :, :] = isqpi*np.sqrt(15./32.)*x*y*2 \
                / (x2+y2+z2+tiny)
        #
        elif (ell == 4) & (m == 4):
            Ylm[j, :, :] = isqpi*3.0/16.0*np.sqrt(35./2.)*(x2*x2-6.0*x2*y2+y4) \
                / (x2+y2+z2+tiny)**2
        elif (ell == 4) & (m == 3):
            Ylm[j, :, :] = isqpi*3.0/8.0*np.sqrt(35.)*(x2-3.0*y2)*z*x \
                / (x2+y2+z2+tiny)**2
        elif (ell == 4) & (m == 2):
            Ylm[j, :, :] = isqpi*3.0/8.0*np.sqrt(5./2.)*(6.0*z2*(x2-y2)-x2*x2+y4) \
                / (x2+y2+z2+tiny)**2
        elif (ell == 4) & (m == 1):
            Ylm[j, :, :] = isqpi*3.0/8.0*np.sqrt(5.)*3.0*(4.0/3.0*z2-x2-y2)*x*z \
                / (x2+y2+z2+tiny)**2
        elif (ell == 4) & (m == 0):
            Ylm[j, :, :] = isqpi*3.0/16.0*np.sqrt(1.)*8.0*(z4-3.0*z2*(x2+y2)+3.0/8.0*(x2*x2+2*x2*y2+y4)) \
                / (x2+y2+z2+tiny)**2
        elif (ell == 4) & (m == -1):
            Ylm[j, :, :] = isqpi*3.0/8.0*np.sqrt(5.)*3.0*(4.0/3.0*z2-x2-y2)*y*z \
                / (x2+y2+z2+tiny)**2
        elif (ell == 4) & (m == -2):
            Ylm[j, :, :] = isqpi*3.0/8.0*np.sqrt(5./2.)*(2.0*x*y*(6.0*z2-x2-y2)) \
                / (x2+y2+z2+tiny)**2
        elif (ell == 4) & (m == -3):
            Ylm[j, :, :] = isqpi*3.0/8.0*np.sqrt(35.)*(3.0*x2*y-y3)*z \
                / (x2+y2+z2+tiny)**2
        elif (ell == 4) & (m == -4):
            Ylm[j, :, :] = isqpi*3.0/16.0*np.sqrt(35./2.)*(4.0*x*(x2*y-y3)) \
                / (x2+y2+z2+tiny)**2
        #
        else:
            print("Illegal Ylm arguments: ", ell, m)
            exit()
    # Done with the loop over x slabs
    return Ylm


def compute_one_ell(dens_N, Nfft_star, ell, grid):
    # Compute the correlation submatrix for a given ell
    # Loop over all m
    msc = grid.max_sep_cell    # Rename for brevity
    corr = np.empty((2*msc+1,) * 3)
    total = np.zeros_like(corr)
    for m in range(-ell, ell+1):
        print("Computing ell, m = %1d %2d." % (ell, m),)
        t = timer()
        Ylm = makeYlm(ell, m, grid.xcell, grid.ycell, grid.zcell)
        Ylm *= dens_N
        print(" NY=%6.4f" % float(timer()-t),)
        t = timer()
        NYfft = np.fft.rfftn(Ylm)*Nfft_star
        Ylm = np.fft.irfftn(NYfft)
        print(" FFT=%6.4f" % float(timer()-t),)
        # Now extract the submatrix
        t = timer()
        corr[msc:, msc:, msc:] = Ylm[0:msc+1, 0:msc+1, 0:msc+1]
        corr[msc:, msc:, :msc] = Ylm[0:msc+1, 0:msc+1,   -msc:]
        corr[msc:, :msc, msc:] = Ylm[0:msc+1,   -msc:, 0:msc+1]
        corr[msc:, :msc, :msc] = Ylm[0:msc+1,   -msc:,   -msc:]
        corr[:msc, msc:, msc:] = Ylm[-msc:, 0:msc+1, 0:msc+1]
        corr[:msc, msc:, :msc] = Ylm[-msc:, 0:msc+1,   -msc:]
        corr[:msc, :msc, msc:] = Ylm[-msc:,   -msc:, 0:msc+1]
        corr[:msc, :msc, :msc] = Ylm[-msc:,   -msc:,   -msc:]
        # We need to multiply it by Ylm of the separation matrix
        corr *= makeYlm(ell, m, grid.corr_cell, grid.corr_cell, grid.corr_cell)
        total += corr
        print(" Sub=%6.4f" % float(timer()-t))
    # Multiplying by 4*pi, to match SE15 equation 13
    total *= 4.0*np.pi
    print(total[msc-1:msc+2, msc-1:msc+2, msc-1:msc+2])
    return total


def hist_corr_grid(corr, rcorr, bins):
    # We're given a list of correlation submatrices 'corr'
    # and their radii 'rcorr'
    # We want to histogram them into the given bin edges 'bins'
    # Also return the number of points in a bin and the bin edges
    hist_corr = []
    for j in range(len(corr)):
        hist_corr.append(np.histogram(rcorr, bins=bins, weights=corr[j])[0])
    corrN, edges = np.histogram(rcorr, bins=bins)
    return hist_corr, corrN, edges


def correlate(pos, w, grid, bins):
    # Run the correlation for the given pos, w pairing
    # Histogram the results onto the given binning
    #
    # Convert the points to the grid
    t = timer()
    grid_N, residual_N = grid.pos2grid(pos)
    print("Done assigning grid points")
    #
    # Form the N densities
    # TODO: If we could align this dens_N array, it would be better!
    dens_N, histbins = np.histogramdd(grid_N, weights=w,
                                      bins=grid.ngrid, range=((0, grid.ngrid), (0, grid.ngrid), (0, grid.ngrid)))
    print("Sum of density grid: ", np.sum(dens_N), )
    print("Type of density grid: ", np.result_type(dens_N), )
    print(" Time=%6.4f" % float(timer()-t))
    #
    # Compute the FFT of the density field, since we use it a lot
    t = timer()
    Nfft_star = np.conj(np.fft.rfftn(dens_N))
    print("Done with FFT(density).  Time=%6.4f" % float(timer()-t))
    # Now correlate!
    corr = []
    for ell in range(0, MAX_ELL+1, 2):
        corr.append(compute_one_ell(dens_N, Nfft_star, ell, grid))
    #
    # Now histogram
    return hist_corr_grid(corr, grid.rcorr, bins)


def linear_binning(max_sep, dsep):
    # Return the edges of separation binning
    # Here we produce linearly space bins
    max_use = dsep*np.floor(1.0*max_sep/dsep)  # Round down
    return np.arange(0, max_use+dsep/2, dsep)


def triplefact(j1, j2, j3):
    """Returns g!/((g - j1)!(g - j2)!(g - j3!)), where g = (j1 + j2 + j3)/2."""
    jhalf = (j1 + j2 + j3) / 2
    return (
        scipy.special.factorial(jhalf) /
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
    return (-1)**(j / 2) * triplefact(j1, j2, j3) / (
        triplefact(2 * j1, 2 * j2, 2 * j3) * (j + 1))**0.5
    # DJE did check this against Wolfram


def Mkl_calc(k, ell, flist):
    """Computes the matrix element from SE15, eq. 9."""
    # This is the matrix element from SE15, eq 9
    jlist = 2 * np.arange(len(flist))  # The f list is j=0,2,4,...
    return (2 * k + 1) * np.sum([
      threej(ell, j, k)**2 * fj for j, fj in zip(jlist[1:], flist[1:])])


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


def setupCPP(pos, w, g, filename):
    with open(filename, "wb") as binfile:
        print(g.ngrid)
        print(g.posmin)
        print(g.boxsize)
        print(g.max_sep)
        binfile.write(struct.pack(
            "dddddddd", *g.posmin, *g.posmax, g.max_sep, 0.0))
        posw = np.empty([len(pos), 4], dtype=np.float64)
        posw[:, 0:3] = pos
        posw[:, 3] = w
        print(posw.shape)
        posw.tofile(binfile)


def write_periodic_random(n, boxsize, filename):
    with open(filename, "wb") as binfile:
        binfile.write(struct.pack("dddddddd",
                                  0.0, 0.0, 0.0,
                                  boxsize, boxsize, boxsize,
                                  0.0, 0.0))
        posw = np.empty([n, 4], dtype=np.float64)
        posw[:, 0:3] = boxsize*np.random.uniform(size=(n, 3))
        posw[:, 3] = np.ones(n)
        posw.tofile(binfile)


#write_periodic_random(100000, 1000.0, "random1e5.box1e3.dat")


def correlateCPP(filename, dsep, ngrid, max_ell, qperiodic, file2=""):
    outfile = filename + ".out"
    s = "%s/fftcorr -in %s -out %s -dr %f -n %d -ell %d" % (
        os.getcwd(), filename, outfile, dsep, ngrid, max_ell)
    if file2:
        s += " -in2 %s" % file2
    if qperiodic:
        s += " -p -r %f" % MAX_SEP
    print(s)
    retcode = subprocess.call(shlex.split(s))
    assert retcode >= 0

    # Load the anisotropic correlation function.
    data = np.loadtxt(outfile)
    data = data[data[:, 0] == 0]  # corr indicated by 0 in first col
    rcen = data[:, 1]
    num = data[:, 2]
    hist_corr = data[:, 3:].T
    return hist_corr, num, rcen


def readCPPoutput(filename):
    f = open(filename, "r")
    P = np.zeros((1000, 5))
    xi = np.zeros((1000, 5))
    Pcnt = 0
    xicnt = 0
    for line in f:
        if ("Anisotropic power" in line):
            Pxi = 1
            continue
        if ("Anisotropic correlation" in line):
            Pxi = 0
            continue
        if ("Estimate of I" in line):
            s = line.rsplit("=", 1)
            I = np.fromstring(s[1], sep=" ")
        if ("divide by I for Pshot" in line):
            s = line.rsplit("=", 1)
            Pshot = np.fromstring(s[1], sep=" ")
        if ("#" in line):
            continue
        # Otherwise, we're going to parse this line
        if (Pxi):
            P[Pcnt, :] = np.fromstring(line, sep=" ")[1:]
            Pcnt += 1
        else:
            xi[xicnt, :] = np.fromstring(line, sep=" ")[1:]
            xicnt += 1
    f.close()
    print("Read %s and found %d P values and %d xi values" %
          (filename, Pcnt, xicnt))
    P = P[:Pcnt]
    xi = xi[:xicnt]
    return xi[:, 2:], xi[:, 1], xi[:, 0], P[:, 2:], P[:, 1], P[:, 0], I, Pshot


def readCPPfast(filename):
    # Just the tables; skip the I and Pshot parsing
    data = np.loadtxt(filename)
    f = np.min(np.where(data[:, 0]) == 0)
    return data[f:, 3:], data[f:, 2], data[f:, 1], data[:f, 3:], data[:f, 2], data[:f, 1], -1.0, -1.0


#####################  Main Code #############
# BOSSpath = "Data/"
# Mockpath = "Patchy/"

#BOSSpath = "/Users/eisenste/cmb/AS2/BOSS/DR12v5/"
#Mockpath = "/Users/eisenste/cmb/AS2/BOSS/DR12v5/Patchy-V6C/"

BOSSpath = "/Users/shallue/sdss/sas/dr12/boss/lss/"
Mockpath = "Patchy/"


def read_galaxies(hemisphere, cosmology, mocks=False):
    ra_rotate = RA_ROTATE[hemisphere]
    if mocks:
        dfile = os.path.join(
            Mockpath, "untar/Patchy-Mocks-DR12CMASS-%s-V6C-Portsmouth-mass_0001.dat" % hemisphere[0])
        rfile = os.path.join(
            Mockpath, "Random-DR12CMASS-%s-V6C-x50.dat.gz" % hemisphere[0])
        D = read_data_file(dfile, "patchy", 0, cosmology, ra_rotate)
        R = read_data_file(rfile, "patchy", 51*len(D.w), cosmology, ra_rotate)
    else:
        dfile = os.path.join(
            BOSSpath, "galaxy_DR12v5_CMASS_%s.fits.gz" % hemisphere)
        rfile = os.path.join(
            BOSSpath, "random0_DR12v5_CMASS_%s.fits.gz" % hemisphere)
        D = read_data_file(dfile, "boss", 0, cosmology, ra_rotate)
        R = read_data_file(rfile, "boss", 51*len(D.w), cosmology, ra_rotate)
    print()
    lapsed_time("io")
    return D, R
    # We're returning [pos,w] pairs, to keep data together


def setup_grid(D, R, ngrid, max_sep):
    # Set up the N=D-R vector and find the bounding box
    # Set the randoms to have negative weight
    R.w *= -np.sum(D.w)/np.sum(R.w)
    N = Catalog(np.concatenate((D.pos, R.pos)), np.concatenate((D.w, R.w)))
    # Now we have the N and R lists to process

    posmin = np.amin(N.pos, axis=0)
    posmax = np.amax(N.pos, axis=0)
    print("Position minima: ", posmin)
    print("Position maxima: ", posmax)
    grid = Grid(ngrid, posmin, posmax, max_sep)

    lapsed_time("setup")
    return N, grid


def writeCPPfiles(D, R, grid, DDfile, RRfile):
    setupCPP(D.pos, D.w, grid, DDfile)
    setupCPP(R.pos, R.w, grid, RRfile)
    print("Wrote CPP binary input files to ", DDfile, "and", RRfile)


def analyze(hist_corrNN, hist_corrRR, rcen):
    #### Analyze the results ####
    fRR = hist_corrRR / hist_corrRR[0]
    xi_raw = hist_corrNN / hist_corrRR[0]
    xi = boundary_correct(xi_raw, fRR)

    xir2_raw = xi_raw*rcen**2
    xir2 = xi*rcen**2

    print()
    print("rcen   xir2[0]    xir2[1]    xir2_raw[0] xir2_raw[1] fRR")
    for j in range(len(rcen)):
        line = ["%6.2f" % rcen[j]]
        for i in range(len(xi)):
            line.append("%10.5f" % xir2[i][j])
        for i in range(len(xi)):
            line.append("%10.5f" % xir2_raw[i][j])
        for i in range(1, len(xi)):
            line.append("%9.6f" % fRR[i][j])
        print(" ".join(line))
    return rcen, xi, xi_raw, fRR


def make_patchy(hemisphere, file_range, cosmology, ngrid, max_sep):
    # Let hemisphere = "North" or "South"
    # Let file_range be a range of numbers, e.g., range(1,10)
    # If no files are given, then do randoms
    ra_rotate = RA_ROTATE[hemisphere]
    # We will use mock 0001 to set the box size
    filename = "untar/Patchy-Mocks-DR12CMASS-%s-V6C-Portsmouth-mass_0001.dat" % (
        hemisphere)
    D = read_data_file(Mockpath+filename, "patchy", 0, cosmology, ra_rotate)
    N, grid = setup_grid(D, D, ngrid, max_sep)

    # Could have had this, but it's annoying to reload the randoms
    # N,grid = setup_grid(D,R,MAX_SEP)

    for f in file_range:
        # Get file %f
        filename = "untar/Patchy-Mocks-DR12CMASS-%s-V6C-Portsmouth-mass_%04d.dat" % (
            hemisphere[0], f)
        D = read_data_file(Mockpath+filename, "patchy",
                           0, cosmology, ra_rotate)
        out = "binary/patchy-DR12CMASS-%s-V6C-%04d.dat" % (hemisphere[0], f)
        setupCPP(D.pos, D.w, grid, Mockpath+out)

    if (len(file_range) == 0):
        filename = "Random-DR12CMASS-%s-V6C-x50.dat.gz" % hemisphere[0]
        R = read_data_file(Mockpath+filename, "patchy",
                           100*len(D.w), cosmology, ra_rotate)
        # Need to renormalize the weights
        # Set the randoms to have negative weight
        R.w *= np.sum(D.w)/np.sum(R.w)
        out = "binary/patchy-DR12CMASS-%s-V6C-random50.dat" % hemisphere[0]
        setupCPP(R.pos, R.w, grid, Mockpath+out)


def run_patchy():
    make_patchy("S", [], COSMOLOGY, NGRID, MAX_SEP)
    make_patchy("N", [], COSMOLOGY, NGRID, MAX_SEP)
    #make_patchy('S', range(1,601))
    #make_patchy('N', range(1,601))

##########################


def analyze_set():
    corrRR, xinum, rcen, PRR, Pnum, kcen, I, _Pshot = \
        readCPPoutput("Output/patchy-DR12CMASS-N-V6C-ran-c3r300.dat")
    corr = np.zeros_like(corrRR)
    P = np.zeros_like(PRR)
    cnt = 0
    for n in range(1, 5):
        if (n == 10):
            continue
        _corr, xinum, rcen, _P, Pnum, kcen, _I, Pshot = \
            readCPPoutput("Output/patchy-DR12CMASS-N-V6C-%04d-c3r300.dat" % n)
        corr += _corr
        P += _P
        cnt += 1
    corr /= cnt
    P /= cnt
    P[:, 0] = (P[:, 0]-Pshot)/I
    P[:, 1:] = (P[:, 1:])/I
    P = P.T
    Del = P*kcen/2.0/np.pi**2
    xi = analyze(corr.T, corrRR.T, rcen)
    for j in range(len(kcen)):
        print("%6.4f " % kcen[j],)
        for i in range(0, len(P)):
            print("%9.2f" % Del[i][j],)
        print()
    return xi, Del


#a = analyze_set()

##########################


def run(run_cpp, setup_cpp):
    DDfile = "/tmp/corrDD.dat"
    RRfile = "/tmp/corrRR.dat"
    if (not run_cpp or setup_cpp):
        ####  Read in the data ####
        D, R = read_galaxies("South", COSMOLOGY)
        max_sep = 0.0 if QPERIODIC else MAX_SEP
        N, grid = setup_grid(D, R, NGRID, max_sep)

        # Write out the CPP files
        if (setup_cpp):
            writeCPPfiles(D, R, grid, DDfile, RRfile)

    #### Run the correlations ####

    if (run_cpp):
        hist_corrNN, hist_corr_num, rcen = correlateCPP(
            DDfile, DSEP, NGRID, MAX_ELL, QPERIODIC, file2=RRfile)
        lapsed_time("corrNN")
        hist_corrRR, hist_corr_num, rcen = correlateCPP(
            RRfile, DSEP, NGRID, MAX_ELL, QPERIODIC)
        lapsed_time("corrRR")
    else:
        # Choose the binning
        bins = linear_binning(grid.max_sep, DSEP)

        print("\nCorrelating NN")
        hist_corrNN, hist_corr_num, hist_edges = correlate(
            N.pos, N.w, grid, bins)
        lapsed_time("corrNN")

        print("Correlating RR")
        hist_corrRR, hist_corr_num, hist_edges = correlate(
            R.pos, R.w, grid, bins)
        lapsed_time("corrRR")

        rcen = (hist_edges[0:-1]+hist_edges[1:])/2.0

    if "io" in times:
        print("\nTime to read the data files: ", times["io"])
    if "setup" in times:
        print("Time to setup the calculation: ", times["setup"])
    if "corrNN" in times:
        print("Time to run the NN correlations: ", times["corrNN"])
    if "corrRR" in times:
        print("Time to run the RR correlations: ", times["corrRR"])
    print()
    return analyze(hist_corrNN, hist_corrRR, rcen), hist_corrNN, hist_corrRR


run_cpp = True  # Whether to run CPP instead of Python code.
setup_cpp = False  # Whether to convert input files to CPP input format.
# result = run(run_cpp, setup_cpp)
