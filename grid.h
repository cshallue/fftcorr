#ifndef GRID_H
#define GRID_H

#include "d12.cpp"
#include "fft_utils.h"
#include "galaxy.h"
#include "histogram.h"
#include "matrix_utils.h"
#include "merge_sort_omp.cpp"
#include "types.h"

class Grid {
 public:
  // Inputs
  int ngrid[3];     // We might prefer a non-cubic box.  The cells are always
                    // cubic!
  Float max_sep;    // How much separation has already been built in.
  Float posmin[3];  // Including the border; we don't support periodic wrapping
                    // in CIC
  Float posmax[3];  // Including the border; we don't support periodic wrapping
                    // in CIC

  // Items to be computed
  Float posrange[3];             // The range of the padded box
  Float cell_size;               // The size of the cubic cells
  Float origin[3];               // The location of the origin in grid units.
  Float *xcell, *ycell, *zcell;  // The cell centers, relative to the origin

  // Storage for the r-space submatrices
  Float sep;     // The range of separations we'll be histogramming
  int csize[3];  // How many cells we must extract as a submatrix to do the
                 // histogramming.
  int csize3;    // The number of submatrix cells
  Float *cx_cell, *cy_cell,
      *cz_cell;  // The cell centers, relative to zero lag.
  Float *rnorm;  // The radius of each cell, in a flattened submatrix.

  // Storage for the k-space submatrices
  Float k_Nyq;   // The Nyquist frequency for our grid.
  Float kmax;    // The maximum wavenumber we'll use
  int ksize[3];  // How many cells we must extract as a submatrix to do the
                 // histogramming.
  int ksize3;    // The number of submatrix cells
  Float *kx_cell, *ky_cell,
      *kz_cell;      // The cell centers, relative to zero lag.
  Float *knorm;      // The wavenumber of each cell, in a flattened submatrix.
  Float *CICwindow;  // The inverse of the window function for the CIC cell
                     // assignment

  // The big grids
  int ngrid2;      // ngrid[2] padded out for the FFT work
  uint64 ngrid3;   // The total number of FFT grid cells
  Float *dens;     // The density field, in a flattened grid
  Float *densFFT;  // The FFT of the density field, in a flattened grid.
  Float *work;     // Work space for each (ell,m), in a flattened grid.

  int cnt;      // The number of galaxies read in.
  Float Pshot;  // The sum of squares of the weights, which is the shot noise
                // for P_0.

  // Positions need to arrive in a coordinate system that has the observer at
  // the origin

  ~Grid() {
    if (dens != NULL) free(dens);
    if (densFFT != NULL) free(densFFT);
    if (work != NULL) free(work);
    free(zcell);
    free(ycell);
    free(xcell);
    free(rnorm);
    free(cx_cell);
    free(cy_cell);
    free(cz_cell);
    free(knorm);
    free(kx_cell);
    free(ky_cell);
    free(kz_cell);
    free(CICwindow);
    // *densFFT and *work are freed in the correlate() routine.
  }

  Grid(const char filename[], int _ngrid[3], Float _cell, Float _sep,
       int qperiodic) {
    // This constructor is rather elaborate, but we're going to do most of the
    // setup. filename and filename2 are the input particles. filename2==NULL
    // will skip that one. _sep is used here simply to adjust the box size if
    // needed. qperiodic flag will configure for periodic BC

    // Have to set these to null so that the initialization will work.
    dens = densFFT = work = NULL;
    rnorm = knorm = CICwindow = NULL;

    // Open a binary input file
    // Setup.Start();
    FILE *fp = fopen(filename, "rb");
    assert(fp != NULL);

    for (int j = 0; j < 3; j++) ngrid[j] = _ngrid[j];
    assert(ngrid[0] > 0 && ngrid[0] < 1e4);
    assert(ngrid[1] > 0 && ngrid[1] < 1e4);
    assert(ngrid[2] > 0 && ngrid[2] < 1e4);

    Float TOOBIG = 1e10;
    // This header is 64 bytes long.
    // Read posmin[3], posmax[3], max_sep, blank8;
    double tmp[4];
    int nread;
    nread = fread(tmp, sizeof(double), 3, fp);
    assert(nread == 3);
    for (int j = 0; j < 3; j++) {
      posmin[j] = tmp[j];
      assert(fabs(posmin[j]) < TOOBIG);
      fprintf(stderr, "posmin[%d] = %f\n", j, posmin[j]);
    }
    nread = fread(tmp, sizeof(double), 3, fp);
    assert(nread == 3);
    for (int j = 0; j < 3; j++) {
      posmax[j] = tmp[j];
      assert(fabs(posmax[j]) < TOOBIG);
      fprintf(stderr, "posmax[%d] = %f\n", j, posmax[j]);
    }
    nread = fread(tmp, sizeof(double), 1, fp);
    assert(nread == 1);
    max_sep = tmp[0];
    assert(max_sep >= 0 && max_sep < TOOBIG);
    fprintf(stderr, "max_sep = %f\n", max_sep);
    nread = fread(tmp, sizeof(double), 1, fp);
    assert(nread == 1);  // Not used, just for alignment
    fclose(fp);

    // If we're going to permute the axes, change here and in
    // add_particles_to_grid(). The answers should be unchanged under
    // permutation std::swap(posmin[0], posmin[1]); std::swap(posmax[0],
    // posmax[1]); std::swap(posmin[2], posmin[1]); std::swap(posmax[2],
    // posmax[1]);

    // If the user wants periodic BC, then we can ignore separation issues.
    if (qperiodic) max_sep = (posmax[0] - posmin[0]) * 100.0;
    fprintf(stderr, "max_sep = %f\n", max_sep);

    // If the user asked for a larger separation than what was planned in the
    // input positions, then we can accomodate.  Add the extra padding to
    // posrange; don't change posmin, since that changes grid registration.
    Float extra_pad = 0.0;
    if (_sep > max_sep) {
      extra_pad = _sep - max_sep;
      max_sep = _sep;
    }
    sep = -1;  // Just as a test that setup() got run

    // Compute the box size required in each direction
    for (int j = 0; j < 3; j++) {
      posmax[j] += extra_pad;
      posrange[j] = posmax[j] - posmin[j];
      assert(posrange[j] > 0.0);
    }

    if (qperiodic || _cell <= 0) {
      // We need to compute the cell size
      // We've been given 3 ngrid and we have the bounding box.
      // Need to pick the most conservative choice
      // This is always required in the periodic case
      cell_size =
          std::max(posrange[0] / ngrid[0],
                   std::max(posrange[1] / ngrid[1], posrange[2] / ngrid[2]));
    } else {
      // We've been given a cell size and a grid.  Need to assure it is ok.
      cell_size = _cell;
      assert(cell_size * ngrid[0] > posrange[0]);
      assert(cell_size * ngrid[1] > posrange[1]);
      assert(cell_size * ngrid[2] > posrange[2]);
    }

    fprintf(stdout, "# Reading file %s.  max_sep=%f\n", filename, max_sep);
    fprintf(stdout, "# Adopting cell_size=%f for ngrid=%d, %d, %d\n", cell_size,
            ngrid[0], ngrid[1], ngrid[2]);
    fprintf(stdout, "# Adopted boxsize: %6.1f %6.1f %6.1f\n",
            cell_size * ngrid[0], cell_size * ngrid[1], cell_size * ngrid[2]);
    fprintf(stdout, "# Input pos range: %6.1f %6.1f %6.1f\n", posrange[0],
            posrange[1], posrange[2]);
    fprintf(stdout, "# Minimum ngrid=%d, %d, %d\n",
            int(ceil(posrange[0] / cell_size)),
            int(ceil(posrange[1] / cell_size)),
            int(ceil(posrange[2] / cell_size)));

    // ngrid2 pads out the array for the in-place FFT.
    // The default 3d FFTW format must have the following:
    ngrid2 = (ngrid[2] / 2 + 1) * 2;  // For the in-place FFT
#ifdef FFTSLAB
// That said, the rest of the code should work even extra space is used.
// Some operations will blindly apply to the pad cells, but that's ok.
// In particular, we might consider having ngrid2 be evenly divisible by
// the critical alignment stride (32 bytes for AVX, but might be more for cache
// lines) or even by a full PAGE for NUMA memory.  Doing this *will* force a
// more complicated FFT, but at least for the NUMA case this is desired: we want
// to force the 2D FFT to run on its socket, and only have the last 1D FFT
// crossing sockets.  Re-using FFTW plans requires the consistent memory
// alignment.
#define FFT_ALIGN 16
    // This is in units of Floats.  16 doubles is 1024 bits.
    ngrid2 = FFT_ALIGN * (ngrid2 / FFT_ALIGN + 1);
#endif
    assert(ngrid2 % 2 == 0);
    fprintf(stdout, "# Using ngrid2=%d for FFT r2c padding\n", ngrid2);
    ngrid3 = (uint64)ngrid[0] * ngrid[1] * ngrid2;

    // Convert origin to grid units
    if (qperiodic) {
      // In this case, we'll place the observer centered in the grid, but
      // then displaced far away in the -x direction
      for (int j = 0; j < 3; j++) origin[j] = ngrid[j] / 2.0;
      origin[0] -= ngrid[0] * 1e6;  // Observer far away!
    } else {
      for (int j = 0; j < 3; j++) origin[j] = (0.0 - posmin[j]) / cell_size;
    }

    // Allocate xcell, ycell, zcell to [ngrid]
    int err;
    err =
        posix_memalign((void **)&xcell, PAGE, sizeof(Float) * ngrid[0] + PAGE);
    assert(err == 0);
    err =
        posix_memalign((void **)&ycell, PAGE, sizeof(Float) * ngrid[1] + PAGE);
    assert(err == 0);
    err =
        posix_memalign((void **)&zcell, PAGE, sizeof(Float) * ngrid[2] + PAGE);
    assert(err == 0);
    assert(xcell != NULL);
    assert(ycell != NULL);
    assert(zcell != NULL);
    // Now set up the cell centers relative to the origin, in grid units
    for (int j = 0; j < ngrid[0]; j++) xcell[j] = 0.5 + j - origin[0];
    for (int j = 0; j < ngrid[1]; j++) ycell[j] = 0.5 + j - origin[1];
    for (int j = 0; j < ngrid[2]; j++) zcell[j] = 0.5 + j - origin[2];
    // Setup.Stop();

    // Allocate dens to [ngrid**2*ngrid2] and set it to zero
    initialize_matrix(dens, ngrid3, ngrid[0]);
    return;
  }

  /* ------------------------------------------------------------------- */

  void read_galaxies(const char filename[], const char filename2[],
                     int qperiodic) {
    // Read to the end of the file, bringing in x,y,z,w points.
    // Bin them onto the grid.
    // We're setting up a large buffer to read in the galaxies.
    // Will reset the buffer periodically, just to limit the size.
    double tmp[8];
    cnt = 0;
    uint64 index;
    Float totw = 0.0, totwsq = 0.0;
// Set up a small buffer, just to reduce the calls to fread, which seem to be
// slow on some machines.
#define BUFFERSIZE 512
    double buffer[BUFFERSIZE], *b;
#define MAXGAL 1000000
    std::vector<Galaxy> gal;
    gal.reserve(
        MAXGAL);  // Just to cut down on thrashing; it will expand as needed

    // IO.Start();
    for (int file = 0; file < 2; file++) {
      char *fn;
      int thiscnt = 0;
      if (file == 0)
        fn = (char *)filename;
      else
        fn = (char *)filename2;
      if (fn == NULL) continue;  // No file!
      fprintf(stdout, "# Reading from file %d named %s\n", file, fn);
      FILE *fp = fopen(fn, "rb");
      assert(fp != NULL);
      int nread = fread(tmp, sizeof(double), 8, fp);
      assert(nread == 8);  // Skip the header
      while ((nread = fread(&buffer, sizeof(double), BUFFERSIZE, fp)) > 0) {
        b = buffer;
        for (int j = 0; j < nread; j += 4, b += 4) {
          index = change_to_grid_coords(b);
          gal.push_back(Galaxy(b, index));
          thiscnt++;
          totw += b[3];
          totwsq += b[3] * b[3];
          if (gal.size() >= MAXGAL) {
            // IO.Stop();
            add_to_grid(gal);
            // IO.Start();
          }
        }
        if (nread != BUFFERSIZE) break;
      }
      cnt += thiscnt;
      fprintf(stdout, "# Found %d galaxies in this file\n", thiscnt);
      fclose(fp);
    }
    // IO.Stop();
    // Add the remaining galaxies to the grid
    add_to_grid(gal);

    fprintf(stdout, "# Found %d particles. Total weight %10.4e.\n", cnt, totw);
    Float totw2 = sum_matrix(dens, ngrid3, ngrid[0]);
    fprintf(stdout, "# Sum of grid is %10.4e (delta = %10.4e)\n", totw2,
            totw2 - totw);
    if (qperiodic == 2) {
      // We're asked to set the mean to zero
      Float mean = totw / ngrid[0] / ngrid[1] / ngrid[2];
      addscalarto_matrix(dens, -mean, ngrid3, ngrid[0]);
      fprintf(stdout, "# Subtracting mean cell density %10.4e\n", mean);
    }

    Float sumsq_dens = sumsq_matrix(dens, ngrid3, ngrid[0]);
    fprintf(stdout, "# Sum of squares of density = %14.7e\n", sumsq_dens);
    Pshot = totwsq;
    fprintf(stdout,
            "# Sum of squares of weights (divide by I for Pshot) = %14.7e\n",
            Pshot);
// When run with N=D-R, this divided by I would be the shot noise.

// Meanwhile, an estimate of I when running with only R is
// (sum of R^2)/Vcell - (11/20)**3*(sum_R w^2)/Vcell
// The latter is correcting the estimate for shot noise
// The 11/20 factor is for triangular cloud in cell.
#ifndef NEAREST_CELL
#ifdef WAVELET
    fprintf(stdout, "# Using D12 wavelet\n");
#else
    totwsq *= 0.55 * 0.55 * 0.55;
    fprintf(stdout, "# Using triangular cloud-in-cell\n");
#endif
#else
    fprintf(stdout, "# Using nearest cell method\n");
#endif
    Float Vcell = cell_size * cell_size * cell_size;
    fprintf(stdout,
            "# Estimate of I (denominator) = %14.7e - %14.7e = %14.7e\n",
            sumsq_dens / Vcell, totwsq / Vcell, (sumsq_dens - totwsq) / Vcell);

    // In the limit of infinite homogeneous particles in a periodic box:
    // If W=sum(w), then each particle has w = W/N.  totwsq = N*(W/N)^2 = W^2/N.
    // Meanwhile, each cell has density (W/N)*(N/Ncell) = W/Ncell.
    // sumsq_dens/Vcell = W^2/(Ncell*Vcell) = W^2/V.
    // Hence the real shot noise is V/N = 1/n.
    return;
  }

  /* ------------------------------------------------------------------- */

  void add_to_grid(std::vector<Galaxy> &gal) {
    // Given a set of Galaxies, add them to the grid and then reset the list
    // CIC.Start();
    const int galsize = gal.size();

#ifdef DEPRICATED
    // This works, but appears to be slower
    for (int j = 0; j < galsize; j++) add_particle_to_grid(gal[j]);
#else
    // If we're parallelizing this, then we need to keep the threads from
    // stepping on each other.  Do this in slabs, but with only every third slab
    // active at any time.

    // Let's sort the particles by x.
    // Need to supply an equal amount of temporary space to merge sort.
    // Do this by another vector.
    std::vector<Galaxy> tmp;
    tmp.reserve(galsize);
    mergesort_parallel_omp(gal.data(), galsize, tmp.data(),
                           omp_get_max_threads());
    // This just falls back to std::sort if omp_get_max_threads==1

    // Now we need to find the starting point of each slab
    // Galaxies between N and N+1 should be in indices [first[N], first[N+1]).
    // That means that first[N] should be the index of the first galaxy to
    // exceed N.
    int first[ngrid[0] + 1], ptr = 0;
    for (int j = 0; j < galsize; j++)
      while (gal[j].x > ptr) first[ptr++] = j;
    for (; ptr <= ngrid[0]; ptr++) first[ptr] = galsize;

    // Now, we'll loop, with each thread in charge of slab x.
    // Not bothering with NUMA issues.  a) Most of the time is spent waiting for
    // memory to respond, not actually piping between processors.  b) Adjacent
    // slabs may not be on the same memory bank anyways.  Keep it simple.
    int slabset = 3;
#ifdef WAVELET
    slabset = WCELLS;
#endif
    for (int mod = 0; mod < slabset; mod++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int x = mod; x < ngrid[0]; x += slabset) {
        // For each slab, insert these particles
        for (int j = first[x]; j < first[x + 1]; j++)
          add_particle_to_grid(gal[j]);
      }
    }
#endif
    gal.clear();
    gal.reserve(MAXGAL);  // Just to check!
    // CIC.Stop();
    return;
  }

  /* ------------------------------------------------------------------- */

  inline uint64 change_to_grid_coords(Float tmp[4]) {
    // Given tmp[4] = x,y,z,w,
    // Modify to put them in box coordinates.
    // We'll have no use for the original coordinates!
    // tmp[3] (w) is unchanged
    tmp[0] = (tmp[0] - posmin[0]) / cell_size;
    tmp[1] = (tmp[1] - posmin[1]) / cell_size;
    tmp[2] = (tmp[2] - posmin[2]) / cell_size;
    uint64 ix = floor(tmp[0]);
    uint64 iy = floor(tmp[1]);
    uint64 iz = floor(tmp[2]);
    return (iz) + ngrid2 * ((iy) + (ix)*ngrid[1]);
  }

  void add_particle_to_grid(Galaxy g) {
    // Add one particle to the density grid.
    // This does a 27-point triangular cloud-in-cell, unless one invokes
    // NEAREST_CELL.
    uint64 index;  // Trying not to assume that ngrid**3 won't spill 32-bits.
    uint64 ix = floor(g.x);
    uint64 iy = floor(g.y);
    uint64 iz = floor(g.z);

// If we're just doing nearest cell.
#ifdef NEAREST_CELL
    index = (iz) + ngrid2 * ((iy) + (ix)*ngrid[1]);
    dens[index] += g.w;
    return;
#endif

#ifdef WAVELET
    // In the wavelet version, we truncate to 1/WAVESAMPLE resolution in each
    // cell and use a lookup table.  Table is set up so that each sub-cell
    // resolution has the values for the various integral cell offsets
    // contiguous in memory.
    uint64 sx = floor((g.x - ix) * WAVESAMPLE);
    uint64 sy = floor((g.y - iy) * WAVESAMPLE);
    uint64 sz = floor((g.z - iz) * WAVESAMPLE);
    const Float *xwave = wave + sx * WCELLS;
    const Float *ywave = wave + sy * WCELLS;
    const Float *zwave = wave + sz * WCELLS;
    // This code does periodic wrapping
    const uint64 ng0 = ngrid[0];
    const uint64 ng1 = ngrid[1];
    const uint64 ng2 = ngrid[2];
    // Offset to the lower-most cell, taking care to handle unsigned int
    ix = (ix + ng0 + WMIN) % ng0;
    iy = (iy + ng1 + WMIN) % ng1;
    iz = (iz + ng2 + WMIN) % ng2;
    Float *px = dens + ngrid2 * ng1 * ix;
    for (int ox = 0; ox < WCELLS; ox++, px += ngrid2 * ng1) {
      if (ix + ox == ng0) px -= ng0 * ng1 * ngrid2;  // Periodic wrap in X
      Float Dx = xwave[ox] * g.w;
      Float *py = px + iy * ngrid2;
      for (int oy = 0; oy < WCELLS; oy++, py += ngrid2) {
        if (iy + oy == ng1) py -= ng1 * ngrid2;  // Periodic wrap in Y
        Float *pz = py + iz;
        Float Dxy = Dx * ywave[oy];
        if (iz + WCELLS > ng2) {  // Z Wrap is needed
          for (int oz = 0; oz < WCELLS; oz++) {
            if (iz + oz == ng2) pz -= ng2;  // Periodic wrap in Z
            pz[oz] += zwave[oz] * Dxy;
          }
        } else {
          for (int oz = 0; oz < WCELLS; oz++) pz[oz] += zwave[oz] * Dxy;
        }
      }
    }
    return;
#endif

    // Now to Cloud-in-Cell
    Float rx = g.x - ix;
    Float ry = g.y - iy;
    Float rz = g.z - iz;
    //
    Float xm = 0.5 * (1 - rx) * (1 - rx) * g.w;
    Float xp = 0.5 * rx * rx * g.w;
    Float x0 = (0.5 + rx - rx * rx) * g.w;
    Float ym = 0.5 * (1 - ry) * (1 - ry);
    Float yp = 0.5 * ry * ry;
    Float y0 = 0.5 + ry - ry * ry;
    Float zm = 0.5 * (1 - rz) * (1 - rz);
    Float zp = 0.5 * rz * rz;
    Float z0 = 0.5 + rz - rz * rz;
    //
    if (ix == 0 || ix == ngrid[0] - 1 || iy == 0 || iy == ngrid[1] - 1 ||
        iz == 0 || iz == ngrid[2] - 1) {
      // This code does periodic wrapping
      const uint64 ng0 = ngrid[0];
      const uint64 ng1 = ngrid[1];
      const uint64 ng2 = ngrid[2];
      ix += ngrid[0];  // Just to put away any fears of negative mods
      iy += ngrid[1];
      iz += ngrid[2];
      const uint64 izm = (iz - 1) % ng2;
      const uint64 iz0 = (iz) % ng2;
      const uint64 izp = (iz + 1) % ng2;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * ym * zm;
      dens[index + iz0] += xm * ym * z0;
      dens[index + izp] += xm * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * y0 * zm;
      dens[index + iz0] += xm * y0 * z0;
      dens[index + izp] += xm * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * yp * zm;
      dens[index + iz0] += xm * yp * z0;
      dens[index + izp] += xm * yp * zp;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * ym * zm;
      dens[index + iz0] += x0 * ym * z0;
      dens[index + izp] += x0 * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * y0 * zm;
      dens[index + iz0] += x0 * y0 * z0;
      dens[index + izp] += x0 * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * yp * zm;
      dens[index + iz0] += x0 * yp * z0;
      dens[index + izp] += x0 * yp * zp;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * ym * zm;
      dens[index + iz0] += xp * ym * z0;
      dens[index + izp] += xp * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * y0 * zm;
      dens[index + iz0] += xp * y0 * z0;
      dens[index + izp] += xp * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * yp * zm;
      dens[index + iz0] += xp * yp * z0;
      dens[index + izp] += xp * yp * zp;
    } else {
      // This code is faster, but doesn't do periodic wrapping
      index = (iz - 1) + ngrid2 * ((iy - 1) + (ix - 1) * ngrid[1]);
      dens[index++] += xm * ym * zm;
      dens[index++] += xm * ym * z0;
      dens[index] += xm * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += xm * y0 * zm;
      dens[index++] += xm * y0 * z0;
      dens[index] += xm * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += xm * yp * zm;
      dens[index++] += xm * yp * z0;
      dens[index] += xm * yp * zp;
      index = (iz - 1) + ngrid2 * ((iy - 1) + ix * ngrid[1]);
      dens[index++] += x0 * ym * zm;
      dens[index++] += x0 * ym * z0;
      dens[index] += x0 * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += x0 * y0 * zm;
      dens[index++] += x0 * y0 * z0;
      dens[index] += x0 * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += x0 * yp * zm;
      dens[index++] += x0 * yp * z0;
      dens[index] += x0 * yp * zp;
      index = (iz - 1) + ngrid2 * ((iy - 1) + (ix + 1) * ngrid[1]);
      dens[index++] += xp * ym * zm;
      dens[index++] += xp * ym * z0;
      dens[index] += xp * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += xp * y0 * zm;
      dens[index++] += xp * y0 * z0;
      dens[index] += xp * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += xp * yp * zm;
      dens[index++] += xp * yp * z0;
      dens[index] += xp * yp * zp;
    }
  }

  /* ------------------------------------------------------------------- */

  Float setup_corr(Float _sep, Float _kmax) {
    // Set up the sub-matrix information, assuming that we'll extract
    // -sep..+sep cells around zero-lag.
    // _sep<0 causes a default to the value in the file.
    // Setup.Start();
    if (_sep < 0)
      sep = max_sep;
    else
      sep = _sep;
    fprintf(stdout, "# Chosen separation %f vs max %f\n", sep, max_sep);
    assert(sep <= max_sep);

    int sep_cell = ceil(sep / cell_size);
    csize[0] = 2 * sep_cell + 1;
    csize[1] = csize[2] = csize[0];
    assert(csize[0] % 2 == 1);
    assert(csize[1] % 2 == 1);
    assert(csize[2] % 2 == 1);
    csize3 = csize[0] * csize[1] * csize[2];
    // Allocate corr_cell to [csize] and rnorm to [csize**3]
    int err;
    err = posix_memalign((void **)&cx_cell, PAGE,
                         sizeof(Float) * csize[0] + PAGE);
    assert(err == 0);
    err = posix_memalign((void **)&cy_cell, PAGE,
                         sizeof(Float) * csize[1] + PAGE);
    assert(err == 0);
    err = posix_memalign((void **)&cz_cell, PAGE,
                         sizeof(Float) * csize[2] + PAGE);
    assert(err == 0);
    initialize_matrix(rnorm, csize3, csize[0]);

    // Normalizing by cell_size just so that the Ylm code can do the wide-angle
    // corrections in the same units.
    for (int i = 0; i < csize[0]; i++) cx_cell[i] = cell_size * (i - sep_cell);
    for (int i = 0; i < csize[1]; i++) cy_cell[i] = cell_size * (i - sep_cell);
    for (int i = 0; i < csize[2]; i++) cz_cell[i] = cell_size * (i - sep_cell);

    for (uint64 i = 0; i < csize[0]; i++)
      for (int j = 0; j < csize[1]; j++)
        for (int k = 0; k < csize[2]; k++)
          rnorm[k + csize[2] * (j + i * csize[1])] =
              cell_size * sqrt((i - sep_cell) * (i - sep_cell) +
                               (j - sep_cell) * (j - sep_cell) +
                               (k - sep_cell) * (k - sep_cell));
    fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n",
            sep_cell);

    // Our box has cubic-sized cells, so k_Nyquist is the same in all directions
    // The spacing of modes is therefore 2*k_Nyq/ngrid
    k_Nyq = M_PI / cell_size;
    kmax = _kmax;
    fprintf(stdout, "# Storing wavenumbers up to %6.4f, with k_Nyq = %6.4f\n",
            kmax, k_Nyq);
    for (int i = 0; i < 3; i++)
      ksize[i] = 2 * ceil(kmax / (2.0 * k_Nyq / ngrid[i])) + 1;
    assert(ksize[0] % 2 == 1);
    assert(ksize[1] % 2 == 1);
    assert(ksize[2] % 2 == 1);
    for (int i = 0; i < 3; i++)
      if (ksize[i] > ngrid[i]) {
        ksize[i] = 2 * floor(ngrid[i] / 2) + 1;
        fprintf(stdout,
                "# WARNING: Requested wavenumber is too big.  Truncating "
                "ksize[%d] to %d\n",
                i, ksize[i]);
      }

    ksize3 = ksize[0] * ksize[1] * ksize[2];
    // Allocate kX_cell to [ksize] and knorm to [ksize**3]
    err = posix_memalign((void **)&kx_cell, PAGE,
                         sizeof(Float) * ksize[0] + PAGE);
    assert(err == 0);
    err = posix_memalign((void **)&ky_cell, PAGE,
                         sizeof(Float) * ksize[1] + PAGE);
    assert(err == 0);
    err = posix_memalign((void **)&kz_cell, PAGE,
                         sizeof(Float) * ksize[2] + PAGE);
    assert(err == 0);
    initialize_matrix(knorm, ksize3, ksize[0]);
    initialize_matrix(CICwindow, ksize3, ksize[0]);

    for (int i = 0; i < ksize[0]; i++)
      kx_cell[i] = (i - ksize[0] / 2) * 2.0 * k_Nyq / ngrid[0];
    for (int i = 0; i < ksize[1]; i++)
      ky_cell[i] = (i - ksize[1] / 2) * 2.0 * k_Nyq / ngrid[1];
    for (int i = 0; i < ksize[2]; i++)
      kz_cell[i] = (i - ksize[2] / 2) * 2.0 * k_Nyq / ngrid[2];

    for (uint64 i = 0; i < ksize[0]; i++)
      for (int j = 0; j < ksize[1]; j++)
        for (int k = 0; k < ksize[2]; k++) {
          knorm[k + ksize[2] * (j + i * ksize[1])] =
              sqrt(kx_cell[i] * kx_cell[i] + ky_cell[j] * ky_cell[j] +
                   kz_cell[k] * kz_cell[k]);
          // For TSC, the square window is 1-sin^2(kL/2)+2/15*sin^4(kL/2)
          Float sinkxL = sin(kx_cell[i] * cell_size / 2.0);
          Float sinkyL = sin(ky_cell[j] * cell_size / 2.0);
          Float sinkzL = sin(kz_cell[k] * cell_size / 2.0);
          sinkxL *= sinkxL;
          sinkyL *= sinkyL;
          sinkzL *= sinkzL;
          Float Wx, Wy, Wz;
          Wx = 1 - sinkxL + 2.0 / 15.0 * sinkxL * sinkxL;
          Wy = 1 - sinkyL + 2.0 / 15.0 * sinkyL * sinkyL;
          Wz = 1 - sinkzL + 2.0 / 15.0 * sinkzL * sinkzL;
          Float window = Wx * Wy * Wz;  // This is the square of the window
#ifdef NEAREST_CELL
          // For this case, the window is unity
          window = 1.0;
#endif
#ifdef WAVELET
          // For this case, the window is unity
          window = 1.0;
#endif
          CICwindow[k + ksize[2] * (j + i * ksize[1])] = 1.0 / window;
          // We will divide the power spectrum by the square of the window
        }

    fprintf(stdout,
            "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
            ksize[0] / 2, ksize[1] / 2, ksize[2] / 2);

    // Setup.Stop();
    return sep;
  }

  void print_submatrix(Float *m, int n, int p, FILE *fp, Float norm) {
    // Print the inner part of a matrix(n,n,n) for debugging
    int mid = n / 2;
    assert(p <= mid);
    for (int i = -p; i <= p; i++)
      for (int j = -p; j <= p; j++) {
        fprintf(fp, "%2d %2d", i, j);
        for (int k = -p; k <= p; k++) {
          // We want to print mid+i, mid+j, mid+k
          fprintf(fp, " %12.8g",
                  m[((mid + i) * n + (mid + j)) * n + mid + k] * norm);
        }
        fprintf(fp, "\n");
      }
    return;
  }

  /* ------------------------------------------------------------------- */

  void correlate(int maxell, Histogram &h, Histogram &kh,
                 int wide_angle_exponent) {
    // Here's where most of the work occurs.
    // This computes the correlations for each ell, summing over m,
    // and then histograms the result.
    void makeYlm(Float * work, int ell, int m, int n[3], int n1, Float *xcell,
                 Float *ycell, Float *zcell, Float *dens, int exponent);

    // Multiply total by 4*pi, to match SE15 normalization
    // Include the FFTW normalization
    Float norm = 4.0 * M_PI / ngrid[0] / ngrid[1] / ngrid[2];
    Float Pnorm = 4.0 * M_PI;
    assert(sep > 0);  // This is a check that the submatrix got set up.

    // Allocate the work matrix and load it with the density
    // We do this here so that the array is touched before FFT planning
    initialize_matrix_by_copy(work, ngrid3, ngrid[0], dens);

    // Allocate total[csize**3] and corr[csize**3]
    Float *total = NULL;
    initialize_matrix(total, csize3, csize[0]);
    Float *corr = NULL;
    initialize_matrix(corr, csize3, csize[0]);
    Float *ktotal = NULL;
    initialize_matrix(ktotal, ksize3, ksize[0]);
    Float *kcorr = NULL;
    initialize_matrix(kcorr, ksize3, ksize[0]);

    /* Setup FFTW */
    fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
    setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, ngrid, ngrid2, work);

    // FFTW might have destroyed the contents of work; need to restore
    // work[]==dens[] So far, I haven't seen this happen.
    if (dens[1] != work[1] || dens[1 + ngrid[2]] != work[1 + ngrid[2]] ||
        dens[ngrid3 - 1] != work[ngrid3 - 1]) {
      fprintf(stdout, "Restoring work matrix\n");
      // Init.Start();
      copy_matrix(work, dens, ngrid3, ngrid[0]);
      // Init.Stop();
    }

    // Correlate .Start();  // Starting the main work
    // Now compute the FFT of the density field and conjugate it
    // FFT(work) in place and conjugate it, storing in densFFT
    fprintf(stdout, "# Computing the density FFT...");
    fflush(NULL);
    FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

    // Correlate.Stop();  // We're tracking initialization separately
    initialize_matrix_by_copy(densFFT, ngrid3, ngrid[0], work);
    fprintf(stdout, "Done!\n");
    fflush(NULL);
    // Correlate.Start();

    // Let's try a check as well -- convert with the 3D code and compare
    /* copy_matrix(work, dens, ngrid3, ngrid[0]);
fftw_execute(fft);
for (uint64 j=0; j<ngrid3; j++)
if (densFFT[j]!=work[j]) {
    int z = j%ngrid2;
    int y = j/ngrid2; y=y%ngrid2;
    int x = j/ngrid[1]/ngrid2;
    printf("%d %d %d  %f  %f\n", x, y, z, densFFT[j], work[j]);
}
*/

    /* ------------ Loop over ell & m --------------- */
    // Loop over each ell to compute the anisotropic correlations
    for (int ell = 0; ell <= maxell; ell += 2) {
      // Initialize the submatrix
      set_matrix(total, 0.0, csize3, csize[0]);
      set_matrix(ktotal, 0.0, ksize3, ksize[0]);
      // Loop over m
      for (int m = -ell; m <= ell; m++) {
        fprintf(stdout, "# Computing %d %2d...", ell, m);
        // Create the Ylm matrix times dens
        makeYlm(work, ell, m, ngrid, ngrid2, xcell, ycell, zcell, dens,
                -wide_angle_exponent);
        fprintf(stdout, "Ylm...");

        // FFT in place
        FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

        // Multiply by conj(densFFT), as complex numbers
        // AtimesB.Start();
        multiply_matrix_with_conjugation((Complex *)work, (Complex *)densFFT,
                                         ngrid3 / 2, ngrid[0]);
        // AtimesB.Stop();

        // Extract the anisotropic power spectrum
        // Load the Ylm's and include the CICwindow correction
        makeYlm(kcorr, ell, m, ksize, ksize[2], kx_cell, ky_cell, kz_cell,
                CICwindow, wide_angle_exponent);
        // Multiply these Ylm by the power result, and then add to total.
        extract_submatrix_C2R(ktotal, kcorr, ksize, (Complex *)work, ngrid,
                              ngrid2);

        // iFFT the result, in place
        IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
        fprintf(stdout, "FFT...");

        // Create Ylm for the submatrix that we'll extract for histogramming
        // The extra multiplication by one here is of negligible cost, since
        // this array is so much smaller than the FFT grid.
        makeYlm(corr, ell, m, csize, csize[2], cx_cell, cy_cell, cz_cell, NULL,
                wide_angle_exponent);

        // Multiply these Ylm by the correlation result, and then add to total.
        extract_submatrix(total, corr, csize, work, ngrid, ngrid2);

        fprintf(stdout, "Done!\n");
      }

      // Extract.Start();
      scale_matrix(total, norm, csize3, csize[0]);
      scale_matrix(ktotal, Pnorm, ksize3, ksize[0]);
      // Extract.Stop();
      // Histogram total by rnorm
      // Hist.Start();
      h.histcorr(ell, csize3, rnorm, total);
      kh.histcorr(ell, ksize3, knorm, ktotal);
      // Hist.Stop();
    }

    /* ------------------- Clean up -------------------*/
    // Free densFFT and Ylm
    free(corr);
    free(total);
    free(kcorr);
    free(ktotal);
    free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

    // Correlate.Stop();
  }
};  // end Grid

#endif  // GRID_H