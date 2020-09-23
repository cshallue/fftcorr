#ifndef GRID_H
#define GRID_H

#include "d12.cpp"
#include "galaxy.h"
#include "histogram.h"
#include "matrix_utils.h"
#include "merge_sort_omp.cpp"
#include "spherical_harmonics.h"
#include "types.h"

class Grid {
 public:
  // Positions need to arrive in a coordinate system that has the observer at
  // the origin_

  ~Grid() {
    if (dens_ != NULL) free(dens_);
    free(zcell_);
    free(ycell_);
    free(xcell_);
    free(rnorm_);
    free(cx_cell_);
    free(cy_cell_);
    free(cz_cell_);
    free(knorm_);
    free(kx_cell_);
    free(ky_cell_);
    free(kz_cell_);
    free(CICwindow_);
  }

  Grid(const char filename[], int ngrid[3], Float cell, Float sep,
       int qperiodic) {
    // This constructor is rather elaborate, but we're going to do most of the
    // setup. filename and filename2 are the input particles. filename2==NULL
    // will skip that one. sep is used here simply to adjust the box size if
    // needed. qperiodic flag will configure for periodic BC

    // Have to set these to null so that the initialization will work.
    dens_ = NULL;
    rnorm_ = knorm_ = CICwindow_ = NULL;

    // Open a binary input file
    // Setup.Start();
    FILE *fp = fopen(filename, "rb");
    assert(fp != NULL);

    for (int j = 0; j < 3; j++) ngrid_[j] = ngrid[j];
    assert(ngrid_[0] > 0 && ngrid_[0] < 1e4);
    assert(ngrid_[1] > 0 && ngrid_[1] < 1e4);
    assert(ngrid_[2] > 0 && ngrid_[2] < 1e4);

    Float TOOBIG = 1e10;
    // This header is 64 bytes long.
    // Read posmin_[3], posmax_[3], max_sep_, blank8;
    double tmp[4];
    int nread;
    nread = fread(tmp, sizeof(double), 3, fp);
    assert(nread == 3);
    for (int j = 0; j < 3; j++) {
      posmin_[j] = tmp[j];
      assert(fabs(posmin_[j]) < TOOBIG);
      fprintf(stderr, "posmin_[%d] = %f\n", j, posmin_[j]);
    }
    nread = fread(tmp, sizeof(double), 3, fp);
    assert(nread == 3);
    for (int j = 0; j < 3; j++) {
      posmax_[j] = tmp[j];
      assert(fabs(posmax_[j]) < TOOBIG);
      fprintf(stderr, "posmax_[%d] = %f\n", j, posmax_[j]);
    }
    nread = fread(tmp, sizeof(double), 1, fp);
    assert(nread == 1);
    max_sep_ = tmp[0];
    assert(max_sep_ >= 0 && max_sep_ < TOOBIG);
    fprintf(stderr, "max_sep_ = %f\n", max_sep_);
    nread = fread(tmp, sizeof(double), 1, fp);
    assert(nread == 1);  // Not used, just for alignment
    fclose(fp);

    // If we're going to permute the axes, change here and in
    // add_particles_to_grid(). The answers should be unchanged under
    // permutation std::swap(posmin_[0], posmin_[1]); std::swap(posmax_[0],
    // posmax_[1]); std::swap(posmin_[2], posmin_[1]); std::swap(posmax_[2],
    // posmax_[1]);

    // If the user wants periodic BC, then we can ignore separation issues.
    if (qperiodic) max_sep_ = (posmax_[0] - posmin_[0]) * 100.0;
    fprintf(stderr, "max_sep_ = %f\n", max_sep_);

    // If the user asked for a larger separation than what was planned in the
    // input positions, then we can accomodate.  Add the extra padding to
    // posrange_; don't change posmin_, since that changes grid registration.
    Float extra_pad = 0.0;
    if (sep > max_sep_) {
      extra_pad = sep - max_sep_;
      max_sep_ = sep;
    }

    // Compute the box size required in each direction
    for (int j = 0; j < 3; j++) {
      posmax_[j] += extra_pad;
      posrange_[j] = posmax_[j] - posmin_[j];
      assert(posrange_[j] > 0.0);
    }

    if (qperiodic || cell <= 0) {
      // We need to compute the cell size
      // We've been given 3 ngrid and we have the bounding box.
      // Need to pick the most conservative choice
      // This is always required in the periodic case
      cell_size_ = std::max(
          posrange_[0] / ngrid_[0],
          std::max(posrange_[1] / ngrid_[1], posrange_[2] / ngrid_[2]));
    } else {
      // We've been given a cell size and a grid.  Need to assure it is ok.
      cell_size_ = cell;
      assert(cell_size_ * ngrid_[0] > posrange_[0]);
      assert(cell_size_ * ngrid_[1] > posrange_[1]);
      assert(cell_size_ * ngrid_[2] > posrange_[2]);
    }

    fprintf(stdout, "# Reading file %s.  max_sep_=%f\n", filename, max_sep_);
    fprintf(stdout, "# Adopting cell_size_=%f for ngrid=%d, %d, %d\n",
            cell_size_, ngrid_[0], ngrid_[1], ngrid_[2]);
    fprintf(stdout, "# Adopted boxsize: %6.1f %6.1f %6.1f\n",
            cell_size_ * ngrid_[0], cell_size_ * ngrid_[1],
            cell_size_ * ngrid_[2]);
    fprintf(stdout, "# Input pos range: %6.1f %6.1f %6.1f\n", posrange_[0],
            posrange_[1], posrange_[2]);
    fprintf(stdout, "# Minimum ngrid=%d, %d, %d\n",
            int(ceil(posrange_[0] / cell_size_)),
            int(ceil(posrange_[1] / cell_size_)),
            int(ceil(posrange_[2] / cell_size_)));

    // ngrid2_ pads out the array for the in-place FFT.
    // The default 3d FFTW format must have the following:
    ngrid2_ = (ngrid_[2] / 2 + 1) * 2;  // For the in-place FFT
#ifdef FFTSLAB
// That said, the rest of the code should work even if extra space is used.
// Some operations will blindly apply to the pad cells, but that's ok.
// In particular, we might consider having ngrid2_ be evenly divisible by
// the critical alignment stride (32 bytes for AVX, but might be more for cache
// lines) or even by a full PAGE for NUMA memory.  Doing this *will* force a
// more complicated FFT, but at least for the NUMA case this is desired: we want
// to force the 2D FFT to run on its socket, and only have the last 1D FFT
// crossing sockets.  Re-using FFTW plans requires the consistent memory
// alignment.
#define FFT_ALIGN 16
    // This is in units of Floats.  16 doubles is 1024 bits.
    ngrid2_ = FFT_ALIGN * (ngrid2_ / FFT_ALIGN + 1);
#endif
    assert(ngrid2_ % 2 == 0);
    fprintf(stdout, "# Using ngrid2_=%d for FFT r2c padding\n", ngrid2_);
    ngrid3_ = (uint64)ngrid_[0] * ngrid_[1] * ngrid2_;

    // Convert origin_ to grid units
    if (qperiodic) {
      // In this case, we'll place the observer centered in the grid, but
      // then displaced far away in the -x direction
      for (int j = 0; j < 3; j++) origin_[j] = ngrid_[j] / 2.0;
      origin_[0] -= ngrid_[0] * 1e6;  // Observer far away!
    } else {
      for (int j = 0; j < 3; j++) origin_[j] = (0.0 - posmin_[j]) / cell_size_;
    }

    // Allocate xcell_, ycell_, zcell_ to [ngrid]
    int err;
    err = posix_memalign((void **)&xcell_, PAGE,
                         sizeof(Float) * ngrid_[0] + PAGE);
    assert(err == 0);
    err = posix_memalign((void **)&ycell_, PAGE,
                         sizeof(Float) * ngrid_[1] + PAGE);
    assert(err == 0);
    err = posix_memalign((void **)&zcell_, PAGE,
                         sizeof(Float) * ngrid_[2] + PAGE);
    assert(err == 0);
    assert(xcell_ != NULL);
    assert(ycell_ != NULL);
    assert(zcell_ != NULL);
    // Now set up the cell centers relative to the origin_, in grid units
    for (int j = 0; j < ngrid_[0]; j++) xcell_[j] = 0.5 + j - origin_[0];
    for (int j = 0; j < ngrid_[1]; j++) ycell_[j] = 0.5 + j - origin_[1];
    for (int j = 0; j < ngrid_[2]; j++) zcell_[j] = 0.5 + j - origin_[2];
    // Setup.Stop();

    // Allocate dens_ to [ngrid**2*ngrid2_] and set it to zero
    initialize_matrix(dens_, ngrid3_, ngrid_[0]);
    return;
  }

  Float cell_size() { return cell_size_; }
  Float ngrid3() { return ngrid3_; }
  Float cnt() { return cnt_; }

  /* ------------------------------------------------------------------- */

  void read_galaxies(const char filename[], const char filename2[],
                     int qperiodic) {
    // Read to the end of the file, bringing in x,y,z,w points.
    // Bin them onto the grid.
    // We're setting up a large buffer to read in the galaxies.
    // Will reset the buffer periodically, just to limit the size.
    double tmp[8];
    cnt_ = 0;
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
      int thiscnt_ = 0;
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
          thiscnt_++;
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
      cnt_ += thiscnt_;
      fprintf(stdout, "# Found %d galaxies in this file\n", thiscnt_);
      fclose(fp);
    }
    // IO.Stop();
    // Add the remaining galaxies to the grid
    add_to_grid(gal);

    fprintf(stdout, "# Found %d particles. Total weight %10.4e.\n", cnt_, totw);
    Float totw2 = sum_matrix(dens_, ngrid3_, ngrid_[0]);
    fprintf(stdout, "# Sum of grid is %10.4e (delta = %10.4e)\n", totw2,
            totw2 - totw);
    if (qperiodic == 2) {
      // We're asked to set the mean to zero
      Float mean = totw / ngrid_[0] / ngrid_[1] / ngrid_[2];
      addscalarto_matrix(dens_, -mean, ngrid3_, ngrid_[0]);
      fprintf(stdout, "# Subtracting mean cell density %10.4e\n", mean);
    }

    Float sumsq_dens_ = sumsq_matrix(dens_, ngrid3_, ngrid_[0]);
    fprintf(stdout, "# Sum of squares of density = %14.7e\n", sumsq_dens_);
    Pshot_ = totwsq;
    fprintf(stdout,
            "# Sum of squares of weights (divide by I for Pshot_) = %14.7e\n",
            Pshot_);
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
    Float Vcell = cell_size_ * cell_size_ * cell_size_;
    fprintf(
        stdout, "# Estimate of I (denominator) = %14.7e - %14.7e = %14.7e\n",
        sumsq_dens_ / Vcell, totwsq / Vcell, (sumsq_dens_ - totwsq) / Vcell);

    // In the limit of infinite homogeneous particles in a periodic box:
    // If W=sum(w), then each particle has w = W/N.  totwsq = N*(W/N)^2 = W^2/N.
    // Meanwhile, each cell has density (W/N)*(N/Ncell) = W/Ncell.
    // sumsq_dens_/Vcell = W^2/(Ncell*Vcell) = W^2/V.
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
    int first[ngrid_[0] + 1], ptr = 0;
    for (int j = 0; j < galsize; j++)
      while (gal[j].x > ptr) first[ptr++] = j;
    for (; ptr <= ngrid_[0]; ptr++) first[ptr] = galsize;

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
      for (int x = mod; x < ngrid_[0]; x += slabset) {
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
    // We'll have no use for the origin_al coordinates!
    // tmp[3] (w) is unchanged
    tmp[0] = (tmp[0] - posmin_[0]) / cell_size_;
    tmp[1] = (tmp[1] - posmin_[1]) / cell_size_;
    tmp[2] = (tmp[2] - posmin_[2]) / cell_size_;
    uint64 ix = floor(tmp[0]);
    uint64 iy = floor(tmp[1]);
    uint64 iz = floor(tmp[2]);
    return (iz) + ngrid2_ * ((iy) + (ix)*ngrid_[1]);
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
    index = (iz) + ngrid2_ * ((iy) + (ix)*ngrid_[1]);
    dens_[index] += g.w;
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
    const uint64 ng0 = ngrid_[0];
    const uint64 ng1 = ngrid_[1];
    const uint64 ng2 = ngrid_[2];
    // Offset to the lower-most cell, taking care to handle unsigned int
    ix = (ix + ng0 + WMIN) % ng0;
    iy = (iy + ng1 + WMIN) % ng1;
    iz = (iz + ng2 + WMIN) % ng2;
    Float *px = dens_ + ngrid2_ * ng1 * ix;
    for (int ox = 0; ox < WCELLS; ox++, px += ngrid2_ * ng1) {
      if (ix + ox == ng0) px -= ng0 * ng1 * ngrid2_;  // Periodic wrap in X
      Float Dx = xwave[ox] * g.w;
      Float *py = px + iy * ngrid2_;
      for (int oy = 0; oy < WCELLS; oy++, py += ngrid2_) {
        if (iy + oy == ng1) py -= ng1 * ngrid2_;  // Periodic wrap in Y
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
    if (ix == 0 || ix == ngrid_[0] - 1 || iy == 0 || iy == ngrid_[1] - 1 ||
        iz == 0 || iz == ngrid_[2] - 1) {
      // This code does periodic wrapping
      const uint64 ng0 = ngrid_[0];
      const uint64 ng1 = ngrid_[1];
      const uint64 ng2 = ngrid_[2];
      ix += ngrid_[0];  // Just to put away any fears of negative mods
      iy += ngrid_[1];
      iz += ngrid_[2];
      const uint64 izm = (iz - 1) % ng2;
      const uint64 iz0 = (iz) % ng2;
      const uint64 izp = (iz + 1) % ng2;
      //
      index = ngrid2_ * (((iy - 1) % ng1) + ((ix - 1) % ng0) * ng1);
      dens_[index + izm] += xm * ym * zm;
      dens_[index + iz0] += xm * ym * z0;
      dens_[index + izp] += xm * ym * zp;
      index = ngrid2_ * (((iy) % ng1) + ((ix - 1) % ng0) * ng1);
      dens_[index + izm] += xm * y0 * zm;
      dens_[index + iz0] += xm * y0 * z0;
      dens_[index + izp] += xm * y0 * zp;
      index = ngrid2_ * (((iy + 1) % ng1) + ((ix - 1) % ng0) * ng1);
      dens_[index + izm] += xm * yp * zm;
      dens_[index + iz0] += xm * yp * z0;
      dens_[index + izp] += xm * yp * zp;
      //
      index = ngrid2_ * (((iy - 1) % ng1) + ((ix) % ng0) * ng1);
      dens_[index + izm] += x0 * ym * zm;
      dens_[index + iz0] += x0 * ym * z0;
      dens_[index + izp] += x0 * ym * zp;
      index = ngrid2_ * (((iy) % ng1) + ((ix) % ng0) * ng1);
      dens_[index + izm] += x0 * y0 * zm;
      dens_[index + iz0] += x0 * y0 * z0;
      dens_[index + izp] += x0 * y0 * zp;
      index = ngrid2_ * (((iy + 1) % ng1) + ((ix) % ng0) * ng1);
      dens_[index + izm] += x0 * yp * zm;
      dens_[index + iz0] += x0 * yp * z0;
      dens_[index + izp] += x0 * yp * zp;
      //
      index = ngrid2_ * (((iy - 1) % ng1) + ((ix + 1) % ng0) * ng1);
      dens_[index + izm] += xp * ym * zm;
      dens_[index + iz0] += xp * ym * z0;
      dens_[index + izp] += xp * ym * zp;
      index = ngrid2_ * (((iy) % ng1) + ((ix + 1) % ng0) * ng1);
      dens_[index + izm] += xp * y0 * zm;
      dens_[index + iz0] += xp * y0 * z0;
      dens_[index + izp] += xp * y0 * zp;
      index = ngrid2_ * (((iy + 1) % ng1) + ((ix + 1) % ng0) * ng1);
      dens_[index + izm] += xp * yp * zm;
      dens_[index + iz0] += xp * yp * z0;
      dens_[index + izp] += xp * yp * zp;
    } else {
      // This code is faster, but doesn't do periodic wrapping
      index = (iz - 1) + ngrid2_ * ((iy - 1) + (ix - 1) * ngrid_[1]);
      dens_[index++] += xm * ym * zm;
      dens_[index++] += xm * ym * z0;
      dens_[index] += xm * ym * zp;
      index += ngrid2_ - 2;  // Step to the next row in y
      dens_[index++] += xm * y0 * zm;
      dens_[index++] += xm * y0 * z0;
      dens_[index] += xm * y0 * zp;
      index += ngrid2_ - 2;  // Step to the next row in y
      dens_[index++] += xm * yp * zm;
      dens_[index++] += xm * yp * z0;
      dens_[index] += xm * yp * zp;
      index = (iz - 1) + ngrid2_ * ((iy - 1) + ix * ngrid_[1]);
      dens_[index++] += x0 * ym * zm;
      dens_[index++] += x0 * ym * z0;
      dens_[index] += x0 * ym * zp;
      index += ngrid2_ - 2;  // Step to the next row in y
      dens_[index++] += x0 * y0 * zm;
      dens_[index++] += x0 * y0 * z0;
      dens_[index] += x0 * y0 * zp;
      index += ngrid2_ - 2;  // Step to the next row in y
      dens_[index++] += x0 * yp * zm;
      dens_[index++] += x0 * yp * z0;
      dens_[index] += x0 * yp * zp;
      index = (iz - 1) + ngrid2_ * ((iy - 1) + (ix + 1) * ngrid_[1]);
      dens_[index++] += xp * ym * zm;
      dens_[index++] += xp * ym * z0;
      dens_[index] += xp * ym * zp;
      index += ngrid2_ - 2;  // Step to the next row in y
      dens_[index++] += xp * y0 * zm;
      dens_[index++] += xp * y0 * z0;
      dens_[index] += xp * y0 * zp;
      index += ngrid2_ - 2;  // Step to the next row in y
      dens_[index++] += xp * yp * zm;
      dens_[index++] += xp * yp * z0;
      dens_[index] += xp * yp * zp;
    }
  }

  /* ------------------------------------------------------------------- */

  // private:
  // Inputs
  int ngrid_[3];     // We might prefer a non-cubic box.  The cells are always
                     // cubic!
  Float max_sep_;    // How much separation has already been built in.
  Float posmin_[3];  // Including the border; we don't support periodic wrapping
                     // in CIC
  Float posmax_[3];  // Including the border; we don't support periodic wrapping
                     // in CIC

  // Items to be computed
  Float posrange_[3];  // The range of the padded box
  Float cell_size_;    // The size of the cubic cells
  Float origin_[3];    // The location of the origin_ in grid units.
  Float *xcell_, *ycell_, *zcell_;  // The cell centers, relative to the origin_

  // Storage for the r-space submatrices
  int csize_[3];  // How many cells we must extract as a submatrix to do the
                  // histogramming.
  int csize3_;    // The number of submatrix cells
  // The cell centers, relative to zero lag.
  Float *cx_cell_, *cy_cell_, *cz_cell_;
  Float *rnorm_;  // The radius of each cell, in a flattened submatrix.

  // Storage for the k-space submatrices
  Float k_Nyq_;   // The Nyquist frequency for our grid.
  Float kmax_;    // The maximum wavenumber we'll use
  int ksize_[3];  // How many cells we must extract as a submatrix to do the
                  // histogramming.
  int ksize3_;    // The number of submatrix cells
  // The cell centers, relative to zero lag.
  Float *kx_cell_, *ky_cell_, *kz_cell_;
  Float *knorm_;      // The wavenumber of each cell, in a flattened submatrix.
  Float *CICwindow_;  // The inverse of the window function for the CIC cell
                      // assignment

  // The big grids
  int ngrid2_;     // ngrid_[2] padded out for the FFT work
  uint64 ngrid3_;  // The total number of FFT grid cells
  Float *dens_;    // The density field, in a flattened grid

  int cnt_;      // The number of galaxies read in.
  Float Pshot_;  // The sum of squares of the weights, which is the shot noise
                 // for P_0.
};               // end Grid

#endif  // GRID_H