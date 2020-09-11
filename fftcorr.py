'''
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

'''


import numpy as np
import wcdm
try:
    import pyfits
except ImportError:
    pass
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


ngrid = 256         # The grid size for the main FFTs
max_sep = 200.0	    # The maximum separation for correlation computation
dsep = 10.0         # The binning of the separations
max_ell = 2         # How many multipoles we'll compute (ell must be even!)
ra_rotate = 44      # The best choice for the BOSS SGC
qperiodic = 1	    # If ==1, use periodic boundary conditions

# And the cosmological parameters
cosmology = {}
cosmology['omega'] = 0.317


##################  Grid class #####################

class Grid:
    '''
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
    '''

    def __init__(self, ngrid, posmin, posmax, max_sep):
        self.ngrid = ngrid
        self.max_sep = max_sep
        self.posmin = posmin
        self.posmax = posmax
        print("Position minima: ", self.posmin)
        print("Position maxima: ", self.posmax)

        # Find the bounding box, including the appropriate buffer
        # Returns the minimum corner of the box and the cubic box size
        self.posmin = self.posmin - self.max_sep/1.5
        self.posmax = self.posmax + self.max_sep/1.5
        self.boxsize = np.max(self.posmax-self.posmin)
        self.cell_size = self.boxsize/self.ngrid
        print("Adopting boxsize %f, cell size %f for a %d^3 grid." %
              (self.boxsize, self.cell_size, self.ngrid))
        # Keep track of the origin on the grid
        tmp1, tmp2 = self.pos2grid(np.zeros(3))
        self.origin = tmp1+tmp2
        print("Origin is at grid location ", self.origin)
        # Make the x,y,z ingredients for the Ylm's
        self.xcell = np.arange(self.ngrid)+0.5-self.origin[0]
        self.ycell = np.arange(self.ngrid)+0.5-self.origin[1]
        self.zcell = np.arange(self.ngrid)+0.5-self.origin[2]

        # Set up the correlation submatrix calculation
        self.max_sep_cell = np.ceil(self.max_sep/self.cell_size)
        print("Correlating to %f will extend to +-%d cells." %
              (self.max_sep, self.max_sep_cell))
        if (self.ngrid < 2*self.max_sep_cell+1):
            print("Grid size is too small relative to separation request")
            exit()
        # This list will be used for the Ylm call
        self.corr_cell = np.arange(-self.max_sep_cell, self.max_sep_cell+1)
        # Make the radial grid
        corr_grid = np.meshgrid(
            self.corr_cell, self.corr_cell, self.corr_cell, indexing='ij')
        self.rcorr = np.sqrt(
            corr_grid[0]**2+corr_grid[1]**2+corr_grid[2]**2)*self.cell_size
        del corr_grid

    def pos2grid(self, pos):
        # Convert positions to grid locations. Also return the residuals
        # Note that this does not check that the resulting grid is within ngrid
        grid = (pos-self.posmin)/self.cell_size
        residual = grid-np.floor(grid)
        return np.floor(grid), residual
# End class Grid


################ Functions ###############################

def coord2pos(ra, dec, rz):
    # Convert angular positions to our Cartesian basis
    global ra_rotate
    return np.array([
        rz*np.cos(dec*np.pi/180.0)*np.cos((ra+ra_rotate)*np.pi/180.0),
        rz*np.cos(dec*np.pi/180.0)*np.sin((ra+ra_rotate)*np.pi/180.0),
        rz*np.sin(dec*np.pi/180.0)
    ]).T


def readdata(filename, Nrandom, cosm, minz=0.43, maxz=0.70):
    # Read a data file.  Use Nrandom to specify the number of randoms to use.
    # Nrandom>0 also triggers treating the weightings as by the random file
    # Return the Cartesian positions and
    print("Reading from %s" % filename)
    hdulist = pyfits.open(filename)
    if (Nrandom > 0):
        data = hdulist[1].data[0:Nrandom]
    else:
        data = hdulist[1].data
    data = data[np.where((data['z'] > minz) & (data['z'] < maxz))]
    print("Done reading and trimming data.")
    redshifts = np.linspace(0.0, maxz+0.1, 1000)
    rz = interpolate.InterpolatedUnivariateSpline(redshifts,
                                                  2997.92*wcdm.coorddist(redshifts, cosm['omega'], -1, 0))
    print("Done computing cosmological distances.")
    pos = coord2pos(data['ra'], data['dec'], rz(data['z']))
    if (Nrandom > 0):
        w = np.float64(data['weight_fkp'])
    else:
        w = np.float64(data['weight_fkp'])*data['weight_systot'] *  \
            (data['weight_cp']+data['weight_noz']-1.0)
    hdulist.close()
    # print(np.result_type(w))
    print("Using %d galaxies, total weight %g" % (len(pos), np.sum(w)))
    result = {}
    result['pos'] = pos
    result['w'] = w
    return result


def read_patchy_file(filename, Nrandom, cosm, minz=0.43, maxz=0.70):
    # Read a data file.  Use Nrandom to specify the number of randoms to use.
    # Nrandom>0 also triggers treating the weightings as by the random file
    # Return the Cartesian positions and
    print("Reading from %s" % filename)
    if (Nrandom > 0):
        data = np.loadtxt(filename,
                          dtype=[('RA', float), ('DEC', float), ('Z', float),
                                 ('NBAR', float), ('BIAS', float), ('VETO', float), ('FIBER', float)])
    else:
        data = np.loadtxt(filename,
                          dtype=[('RA', float), ('DEC', float), ('Z', float), ('MASS', float),
                                 ('NBAR', float), ('BIAS', float), ('VETO', float), ('FIBER', float)])
    if (Nrandom > 0):
        if (Nrandom < len(data)):
            data = data[0:Nrandom]
    data = data[np.where((data['Z'] > minz) & (data['Z'] < maxz))]
    print("Done reading and trimming data.")

    redshifts = np.linspace(0.0, maxz+0.1, 1000)
    rz = interpolate.InterpolatedUnivariateSpline(redshifts,
                                                  2997.92*wcdm.coorddist(redshifts, cosm['omega'], -1, 0))
    print("Done computing cosmological distances.")
    pos = coord2pos(data['RA'], data['DEC'], rz(data['Z']))
    w = data['VETO']*data['FIBER']/(1+1e4*data['NBAR'])
    # print(np.result_type(w))
    print("Using %d galaxies, total weight %g" % (len(pos), np.sum(w)))
    result = {}
    result['pos'] = pos
    result['w'] = w
    return result


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
    y, z = np.meshgrid(ycell, zcell, indexing='ij')
    y2 = y*y     # These get used for each x, so good to cache
    y3 = y2*y     # These get used for each x, so good to cache
    y4 = y3*y     # These get used for each x, so good to cache
    z2 = z*z
    z3 = z2*z
    z4 = z3*z
    # TODO: If we could align this array, it would be better!
    Ylm = np.empty(np.ones(3)*ngrid)
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
    corr = np.empty(np.ones(3)*(2*msc+1))
    total = np.zeros(np.ones(3)*(2*msc+1))
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
    for ell in range(0, max_ell+1, 2):
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
    jhalf = (j1+j2+j3)/2.0
    return (
        scipy.special.factorial(jhalf) / 
        scipy.special.factorial(jhalf-j1) / 
        scipy.special.factorial(jhalf-j2) / 
        scipy.special.factorial(jhalf-j3))


def threej(j1, j2, j3):
    # Compute {j1,j2,j3; 0,0,0} 3j symbol
    j = j1+j2+j3
    if (j % 2 > 0):
        return 0     # Must be even
    if (j1+j2 < j3):
        return 0  # Check legal triangle
    if (j2+j3 < j1):
        return 0  # Check legal triangle
    if (j3+j1 < j2):
        return 0  # Check legal triangle
    return (-1)**(j/2.0) * triplefact(j1, j2, j3) / (triplefact(2*j1, 2*j2, 2*j3)*(j+1))**0.5
    # DJE did check this against Wolfram


def Mkl_calc(k, ell, flist):
    # This is the matrix element from SE15, eq 9
    s = 0.
    for j in np.arange(1, len(flist)):
        # Recall that the f list is ell=0,2,4,...
        s += (threej(ell, 2*j, k))**2*flist[j]
    s *= (2.*k+1.)
    return s


def boundary_correct(xi_raw, fRR):
    xi = np.zeros(xi_raw.shape)
    for r in range(len(fRR[0])):
        Mkl = np.identity(len(fRR))
        for k in range(len(fRR)):
            for l in range(len(fRR)):
                # Remember that the ell's are indexed 0,2,4...
                Mkl[k][l] += Mkl_calc(2*k, 2*l, fRR[:, r])
        if (r == len(fRR[0])-1):
            print(r, Mkl)
        Minv = np.linalg.inv(Mkl)
        xi[:, r] = np.dot(Minv, xi_raw[:, r])
        # TODO: Need to check whether this matrix should have been transposed
    return xi


def setupCPP(pos, w, g, filename):
    binfile = open(filename, "wb")
    print(g.ngrid)
    print(g.posmin)
    print(g.boxsize)
    print(g.max_sep)
    binfile.write(struct.pack("dddddddd",
                              g.posmin[0], g.posmin[1], g.posmin[2],
                              g.posmax[0], g.posmax[1], g.posmax[2],
                              g.max_sep, 0.0))
    posw = np.empty([len(pos), 4], dtype=np.float64)
    posw[:, 0:3] = pos
    posw[:, 3] = w
    print(posw.shape)
    posw.tofile(binfile)
    binfile.close()


def write_periodic_random(n, boxsize, filename):
    binfile = open(filename, "wb")
    binfile.write(struct.pack("dddddddd",
                              0.0, 0.0, 0.0,
                              boxsize, boxsize, boxsize,
                              0.0, 0.0))
    posw = np.empty([n, 4], dtype=np.float64)
    posw[:, 0] = boxsize*np.random.rand(n)
    posw[:, 1] = boxsize*np.random.rand(n)
    posw[:, 2] = boxsize*np.random.rand(n)
    posw[:, 3] = np.ones(n)
    posw.tofile(binfile)
    binfile.close()


#write_periodic_random(100000, 1000.0, "random1e5.box1e3.dat")


def correlateCPP(filename, dsep, file2=""):

    s = "%s/fftcorr -in %s -out %s.out -dr %f -n %d -ell %d" % (
        os.getcwd(), filename, filename, dsep, ngrid, max_ell)
    if (file2 != ""):
        s += " -in2 %s" % file2
    if (qperiodic):
        s += " -p -r %f" % max_sep
    print(s)
    retcode = subprocess.call(shlex.split(s))
    assert retcode >= 0
    data = np.loadtxt(filename+".out")
    return data[:, 2:].T, data[:, 1], data[:, 0]


def readCPPoutput(filename):
    f = open(filename, "r")
    P = np.zeros((1000, 5))
    xi = np.zeros((1000, 5))
    Pcnt = 0
    xicnt = 0
    for line in f:
        if ('Anisotropic power' in line):
            Pxi = 1
            continue
        if ('Anisotropic correlation' in line):
            Pxi = 0
            continue
        if ('Estimate of I' in line):
            s = line.rsplit('=', 1)
            I = np.fromstring(s[1], sep=' ')
        if ('divide by I for Pshot' in line):
            s = line.rsplit('=', 1)
            Pshot = np.fromstring(s[1], sep=' ')
        if ('#' in line):
            continue
        # Otherwise, we're going to parse this line
        if (Pxi):
            P[Pcnt, :] = np.fromstring(line, sep=' ')[1:]
            Pcnt += 1
        else:
            xi[xicnt, :] = np.fromstring(line, sep=' ')[1:]
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
# BOSSpath = 'Data/'
# Mockpath = 'Patchy/'

#BOSSpath = '/Users/eisenste/cmb/AS2/BOSS/DR12v5/'
#Mockpath = '/Users/eisenste/cmb/AS2/BOSS/DR12v5/Patchy-V6C/'

BOSSpath = '/Users/shallue/sdss/sas/dr12/boss/lss/'
Mockpath = 'Patchy/'


def read_dataNGC(cosmology):
    global ra_rotate
    ra_rotate = -142.5
    D = readdata(BOSSpath+'galaxy_DR12v5_CMASS_North.fits.gz', 0, cosmology)
    R = readdata(BOSSpath+'random0_DR12v5_CMASS_North.fits.gz',
                 51*len(D['w']), cosmology)
    print()
    lapsed_time('io')
    return D, R


def read_dataSGC(cosmology):
    global ra_rotate
    ra_rotate = 44
    D = readdata(BOSSpath+'galaxy_DR12v5_CMASS_South.fits.gz', 0, cosmology)
    R = readdata(BOSSpath+'random0_DR12v5_CMASS_South.fits.gz',
                 51*len(D['w']), cosmology)
    print()
    lapsed_time('io')
    return D, R


def read_patchy(cosmology, NGC):
    global ra_rotate
    if (NGC == 'N'):
        ra_rotate = -142.5
        D = read_patchy_file(
            Mockpath+'untar/Patchy-Mocks-DR12CMASS-N-V6C-Portsmouth-mass_0001.dat', 0, cosmology)
        R = read_patchy_file(
            Mockpath+'Random-DR12CMASS-N-V6C-x50.dat.gz', 51*len(D['w']), cosmology)
    else:
        ra_rotate = 44
        D = read_patchy_file(
            Mockpath+'untar/Patchy-Mocks-DR12CMASS-S-V6C-Portsmouth-mass_0001.dat', 0, cosmology)
        R = read_patchy_file(
            Mockpath+'Random-DR12CMASS-S-V6C-x50.dat.gz', 51*len(D['w']), cosmology)
    print()
    lapsed_time('io')
    return D, R
    # We're returning [pos,w] pairs, to keep data together


def setup_grid(D, R, max_sep):
    # Set up the N=D-R vector and find the bounding box
    # Set the randoms to have negative weight
    R['w'] *= -np.sum(D['w'])/np.sum(R['w'])
    N = {}
    N['pos'] = np.concatenate((D['pos'], R['pos']))
    N['w'] = np.concatenate((D['w'], R['w']))
    # Now we have the N and R lists to process

    posmin = np.amin(N['pos'], axis=0)
    posmax = np.amax(N['pos'], axis=0)
    max_sep = 0.0 if qperiodic else max_sep
    grid = Grid(ngrid, posmin, posmax, max_sep)

    lapsed_time('setup')
    return N, grid


def writeCPPfiles(D, R, grid, DDfile, RRfile):
    setupCPP(D['pos'], D['w'], grid, DDfile)
    setupCPP(R['pos'], R['w'], grid, RRfile)
    print("Wrote CPP binary input files to ", DDfile, "and", RRfile)


def analyze(hist_corrNN, hist_corrRR, rcen):
    #### Analyze the results ####
    fRR = np.empty([len(hist_corrRR), len(hist_corrRR[0])])
    xi_raw = np.empty([len(hist_corrRR), len(hist_corrRR[0])])
    for j in range(len(hist_corrRR)):
        fRR[j, :] = hist_corrRR[j]/(hist_corrRR[0]+1e-30)
        xi_raw[j, :] = hist_corrNN[j]/(hist_corrRR[0]+1e-30)
    xi = boundary_correct(xi_raw, fRR)

    xir2_raw = xi_raw*rcen**2
    xir2 = xi*rcen**2

    print()
    for j in range(len(rcen)):
        print("%6.2f" % rcen[j],)
        for i in range(len(xi)):
            print("%10.5f" % xir2[i][j],)
        print("  ",)
        for i in range(len(xi)):
            print("%10.5f" % xir2_raw[i][j],)
        print("  ",)
        for i in range(1, len(xi)):
            print("%9.6f" % fRR[i][j],)
        print()
    return rcen, xi, xi_raw, fRR


def make_patchy(NGC, file_range):
    # Let NGC = 'N' or 'S'
    # Let file_range be a range of numbers, e.g., range(1,10)
    # If no files are given, then do randoms
    global ra_rotate
    if (NGC == 'N'):
        ra_rotate = -142.5
    else:
        ra_rotate = 44
    # We will use mock 0001 to set the box size
    filename = 'untar/Patchy-Mocks-DR12CMASS-%s-V6C-Portsmouth-mass_0001.dat' % (
        NGC)
    D = read_patchy_file(Mockpath+filename, 0, cosmology)
    N, grid = setup_grid(D, D, max_sep)

    # Could have had this, but it's annoying to reload the randoms
    # N,grid = setup_grid(D,R,max_sep)

    for f in file_range:
        # Get file %f
        filename = 'untar/Patchy-Mocks-DR12CMASS-%s-V6C-Portsmouth-mass_%04d.dat' % (
            NGC, f)
        D = read_patchy_file(Mockpath+filename, 0, cosmology)
        out = 'binary/patchy-DR12CMASS-%s-V6C-%04d.dat' % (NGC, f)
        setupCPP(D['pos'], D['w'], grid, Mockpath+out)

    if (len(file_range) == 0):
        filename = 'Random-DR12CMASS-%s-V6C-x50.dat.gz' % (NGC)
        R = read_patchy_file(Mockpath+filename, 100*len(D['w']), cosmology)
        # Need to renormalize the weights
        # Set the randoms to have negative weight
        R['w'] *= np.sum(D['w'])/np.sum(R['w'])
        out = 'binary/patchy-DR12CMASS-%s-V6C-random50.dat' % (NGC)
        setupCPP(R['pos'], R['w'], grid, Mockpath+out)


def run_patchy():
    make_patchy('S', [])
    make_patchy('N', [])
    #make_patchy('S', range(1,601))
    #make_patchy('N', range(1,601))

##########################


def analyze_set():
    corrRR, xinum, rcen, PRR, Pnum, kcen, I, _Pshot = \
        readCPPoutput("Output/patchy-DR12CMASS-N-V6C-ran-c3r300.dat")
    corr = np.zeros(corrRR.shape)
    P = np.zeros(PRR.shape)
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
        #D,R = read_dataNGC(cosmology)
        D, R = read_dataSGC(cosmology)
        N, grid = setup_grid(D, R, max_sep)

        # Write out the CPP files
        if (setup_cpp):
            writeCPPfiles(D, R, grid, DDfile, RRfile)

    #### Run the correlations ####

    if (run_cpp):
        hist_corrNN, hist_corr_num, rcen = correlateCPP(
            DDfile, dsep, file2=RRfile)
        lapsed_time('corrNN')
        hist_corrRR, hist_corr_num, rcen = correlateCPP(RRfile, dsep)
        lapsed_time('corrRR')
    else:
        # Choose the binning
        bins = linear_binning(grid.max_sep, dsep)

        print("\nCorrelating NN")
        hist_corrNN, hist_corr_num, hist_edges = correlate(
            N['pos'], N['w'], grid, bins)
        lapsed_time('corrNN')

        print("Correlating RR")
        hist_corrRR, hist_corr_num, hist_edges = correlate(
            R['pos'], R['w'], grid, bins)
        lapsed_time('corrRR')

        rcen = (hist_edges[0:-1]+hist_edges[1:])/2.0

    if 'io' in times:
        print("\nTime to read the data files: ", times['io'])
    if 'setup' in times:
        print("Time to setup the calculation: ", times['setup'])
    if 'corrNN' in times:
        print("Time to run the NN correlations: ", times['corrNN'])
    if 'corrRR' in times:
        print("Time to run the RR correlations: ", times['corrRR'])
    print()
    return analyze(hist_corrNN, hist_corrRR, rcen), hist_corrNN, hist_corrRR


run_cpp = True  # Whether to run CPP instead of Python code.
setup_cpp = False  # Whether to convert input files to CPP input format.
# result = run(run_cpp, setup_cpp)
