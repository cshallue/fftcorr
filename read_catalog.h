#ifndef READ_GALAXIES_H
#define READ_GALAXIES_H

#include <assert.h>

#include "grid.h"
#include "types.h"

class SurveyBox {
 public:
  SurveyBox() {
    max_sep_ = 0;
    for (int i = 0; i < 3; ++i) {
      posmin_[i] = 0;
      posmax_[i] = 0;
    }
  }

  void read_header(const char filename[]) {
    // Read posmin_[3], posmax_[3], max_sep_, blank8;
    FILE *fp = fopen(filename, "rb");
    assert(fp != NULL);
    double buf[8];
    int nread = fread(buf, sizeof(double), 8, fp);
    assert(nread == 8);
    fclose(fp);

    Float TOOBIG = 1e10;
    for (int i = 0; i < 7; i++) {
      assert(fabs(buf[i]) < TOOBIG);
    }

    posmin_[0] = buf[0];
    posmin_[1] = buf[1];
    posmin_[2] = buf[2];
    posmax_[0] = buf[3];
    posmax_[1] = buf[4];
    posmax_[2] = buf[5];
    max_sep_ = buf[6];
    // buf[7] not used, just for alignment.

    assert(max_sep_ >= 0);

    fprintf(stderr, "Reading survey box from header of %s\n", filename);
    fprintf(stderr, "posmin = [%.4f, %.4f, %.4f]\n", posmin_[0], posmin_[1],
            posmin_[2]);
    fprintf(stderr, "posmax = [%.4f, %.4f, %.4f]\n", posmax_[0], posmax_[1],
            posmax_[2]);
    fprintf(stderr, "max_sep = %f\n", max_sep_);
  }

  // If the user wants periodic BC, then we can ignore separation issues.
  void set_periodic_boundary() { max_sep_ = (posmax_[0] - posmin_[0]) * 100; }

  Float max_sep() { return max_sep_; }
  Float *posmin() { return posmin_; }
  Float *posmax() { return posmax_; }

 private:
  Float max_sep_;
  Float posmin_[3];
  Float posmax_[3];
};

class CatalogReader {
 public:
  /* ------------------------------------------------------------------- */

  void read_galaxies(Grid *grid, const char filename[], const char filename2[],
                     bool zero_center) {
    // filename and filename2 are the input particles. filename2==NULL
    // will skip that one
    // Read to the end of the file, bringing in x,y,z,w points.
    // Bin them onto the grid.
    // We're setting up a large buffer to read in the galaxies.
    // Will reset the buffer periodically, just to limit the size.
    double tmp[8];
    count_ = 0;
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
      int thiscount_ = 0;
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
          index = grid->change_to_grid_coords(b);
          gal.push_back(Galaxy(b, index));
          thiscount_++;
          totw += b[3];
          totwsq += b[3] * b[3];
          if (gal.size() >= MAXGAL) {
            // IO.Stop();
            add_to_grid(grid, gal);
            // IO.Start();
          }
        }
        if (nread != BUFFERSIZE) break;
      }
      count_ += thiscount_;
      fprintf(stdout, "# Found %d galaxies in this file\n", thiscount_);
      fclose(fp);
    }
    // IO.Stop();
    // Add the remaining galaxies to the grid
    add_to_grid(grid, gal);

    fprintf(stdout, "# Found %d particles. Total weight %10.4e.\n", count_,
            totw);
    Float totw2 = sum_matrix(grid->dens(), grid->ngrid3(), grid->ngrid()[0]);
    fprintf(stdout, "# Sum of grid is %10.4e (delta = %10.4e)\n", totw2,
            totw2 - totw);
    if (zero_center) {
      // We're asked to set the mean to zero
      Float mean =
          totw / grid->ngrid()[0] / grid->ngrid()[1] / grid->ngrid()[2];
      addscalarto_matrix(grid->dens_, -mean, grid->ngrid3(), grid->ngrid()[0]);
      fprintf(stdout, "# Subtracting mean cell density %10.4e\n", mean);
    }

    Float sumsq_dens =
        sumsq_matrix(grid->dens(), grid->ngrid3(), grid->ngrid()[0]);
    fprintf(stdout, "# Sum of squares of density = %14.7e\n", sumsq_dens);
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
    Float Vcell = grid->cell_size() * grid->cell_size() * grid->cell_size();
    fprintf(stdout,
            "# Estimate of I (denominator) = %14.7e - %14.7e = %14.7e\n",
            sumsq_dens / Vcell, totwsq / Vcell, (sumsq_dens - totwsq) / Vcell);

    // In the limit of infinite homogeneous particles in a periodic box:
    // If W=sum(w), then each particle has w = W/N.  totwsq = N*(W/N)^2 =
    // W^2/N. Meanwhile, each cell has density (W/N)*(N/Ncell) = W/Ncell.
    // sumsq_dens/Vcell = W^2/(Ncell*Vcell) = W^2/V.
    // Hence the real shot noise is V/N = 1/n.
    return;
  }

  /* ------------------------------------------------------------------- */

  void add_to_grid(Grid *grid, std::vector<Galaxy> &gal) {
    // Given a set of Galaxies, add them to the grid and then reset the list
    // CIC.Start();
    const int galsize = gal.size();

#ifdef DEPRICATED
    // This works, but appears to be slower
    for (int j = 0; j < galsize; j++) add_particle_to_grid(grid, gal[j]);
#else
    // If we're parallelizing this, then we need to keep the threads from
    // stepping on each other.  Do this in slabs, but with only every third
    // slab active at any time.

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
    int first[grid->ngrid()[0] + 1], ptr = 0;
    for (int j = 0; j < galsize; j++)
      while (gal[j].x > ptr) first[ptr++] = j;
    for (; ptr <= grid->ngrid()[0]; ptr++) first[ptr] = galsize;

    // Now, we'll loop, with each thread in charge of slab x.
    // Not bothering with NUMA issues.  a) Most of the time is spent waiting
    // for memory to respond, not actually piping between processors.  b)
    // Adjacent slabs may not be on the same memory bank anyways.  Keep it
    // simple.
    int slabset = 3;
#ifdef WAVELET
    slabset = WCELLS;
#endif
    for (int mod = 0; mod < slabset; mod++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int x = mod; x < grid->ngrid()[0]; x += slabset) {
        // For each slab, insert these particles
        for (int j = first[x]; j < first[x + 1]; j++)
          add_particle_to_grid(grid, gal[j]);
      }
    }
#endif
    gal.clear();
    gal.reserve(MAXGAL);  // Just to check!
    // CIC.Stop();
    return;
  }

  /* ------------------------------------------------------------------- */

  void add_particle_to_grid(Grid *grid, Galaxy g) {
    Float *dens = grid->dens_;

    // Add one particle to the density grid.
    // This does a 27-point triangular cloud-in-cell, unless one invokes
    // NEAREST_CELL.
    uint64 index;  // Trying not to assume that ngrid**3 won't spill 32-bits.
    uint64 ix = floor(g.x);
    uint64 iy = floor(g.y);
    uint64 iz = floor(g.z);

// If we're just doing nearest cell.
#ifdef NEAREST_CELL
    index = (iz) + grid->ngrid2() * ((iy) + (ix)*grid->ngrid()[1]);
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
    const uint64 ng0 = grid->ngrid()[0];
    const uint64 ng1 = grid->ngrid()[1];
    const uint64 ng2 = grid->ngrid()[2];
    // Offset to the lower-most cell, taking care to handle unsigned int
    ix = (ix + ng0 + WMIN) % ng0;
    iy = (iy + ng1 + WMIN) % ng1;
    iz = (iz + ng2 + WMIN) % ng2;
    Float *px = dens + grid->ngrid2() * ng1 * ix;
    for (int ox = 0; ox < WCELLS; ox++, px += grid->ngrid2() * ng1) {
      if (ix + ox == ng0)
        px -= ng0 * ng1 * grid->ngrid2();  // Periodic wrap in X
      Float Dx = xwave[ox] * g.w;
      Float *py = px + iy * grid->ngrid2();
      for (int oy = 0; oy < WCELLS; oy++, py += grid->ngrid2()) {
        if (iy + oy == ng1) py -= ng1 * grid->ngrid2();  // Periodic wrap in Y
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
    if (ix == 0 || ix == grid->ngrid()[0] - 1 || iy == 0 ||
        iy == grid->ngrid()[1] - 1 || iz == 0 || iz == grid->ngrid()[2] - 1) {
      // This code does periodic wrapping
      const uint64 ng0 = grid->ngrid()[0];
      const uint64 ng1 = grid->ngrid()[1];
      const uint64 ng2 = grid->ngrid()[2];
      ix += grid->ngrid()[0];  // Just to put away any fears of negative mods
      iy += grid->ngrid()[1];
      iz += grid->ngrid()[2];
      const uint64 izm = (iz - 1) % ng2;
      const uint64 iz0 = (iz) % ng2;
      const uint64 izp = (iz + 1) % ng2;
      //
      index = grid->ngrid2() * (((iy - 1) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * ym * zm;
      dens[index + iz0] += xm * ym * z0;
      dens[index + izp] += xm * ym * zp;
      index = grid->ngrid2() * (((iy) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * y0 * zm;
      dens[index + iz0] += xm * y0 * z0;
      dens[index + izp] += xm * y0 * zp;
      index = grid->ngrid2() * (((iy + 1) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * yp * zm;
      dens[index + iz0] += xm * yp * z0;
      dens[index + izp] += xm * yp * zp;
      //
      index = grid->ngrid2() * (((iy - 1) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * ym * zm;
      dens[index + iz0] += x0 * ym * z0;
      dens[index + izp] += x0 * ym * zp;
      index = grid->ngrid2() * (((iy) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * y0 * zm;
      dens[index + iz0] += x0 * y0 * z0;
      dens[index + izp] += x0 * y0 * zp;
      index = grid->ngrid2() * (((iy + 1) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * yp * zm;
      dens[index + iz0] += x0 * yp * z0;
      dens[index + izp] += x0 * yp * zp;
      //
      index = grid->ngrid2() * (((iy - 1) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * ym * zm;
      dens[index + iz0] += xp * ym * z0;
      dens[index + izp] += xp * ym * zp;
      index = grid->ngrid2() * (((iy) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * y0 * zm;
      dens[index + iz0] += xp * y0 * z0;
      dens[index + izp] += xp * y0 * zp;
      index = grid->ngrid2() * (((iy + 1) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * yp * zm;
      dens[index + iz0] += xp * yp * z0;
      dens[index + izp] += xp * yp * zp;
    } else {
      // This code is faster, but doesn't do periodic wrapping
      index =
          (iz - 1) + grid->ngrid2() * ((iy - 1) + (ix - 1) * grid->ngrid()[1]);
      dens[index++] += xm * ym * zm;
      dens[index++] += xm * ym * z0;
      dens[index] += xm * ym * zp;
      index += grid->ngrid2() - 2;  // Step to the next row in y
      dens[index++] += xm * y0 * zm;
      dens[index++] += xm * y0 * z0;
      dens[index] += xm * y0 * zp;
      index += grid->ngrid2() - 2;  // Step to the next row in y
      dens[index++] += xm * yp * zm;
      dens[index++] += xm * yp * z0;
      dens[index] += xm * yp * zp;
      index = (iz - 1) + grid->ngrid2() * ((iy - 1) + ix * grid->ngrid()[1]);
      dens[index++] += x0 * ym * zm;
      dens[index++] += x0 * ym * z0;
      dens[index] += x0 * ym * zp;
      index += grid->ngrid2() - 2;  // Step to the next row in y
      dens[index++] += x0 * y0 * zm;
      dens[index++] += x0 * y0 * z0;
      dens[index] += x0 * y0 * zp;
      index += grid->ngrid2() - 2;  // Step to the next row in y
      dens[index++] += x0 * yp * zm;
      dens[index++] += x0 * yp * z0;
      dens[index] += x0 * yp * zp;
      index =
          (iz - 1) + grid->ngrid2() * ((iy - 1) + (ix + 1) * grid->ngrid()[1]);
      dens[index++] += xp * ym * zm;
      dens[index++] += xp * ym * z0;
      dens[index] += xp * ym * zp;
      index += grid->ngrid2() - 2;  // Step to the next row in y
      dens[index++] += xp * y0 * zm;
      dens[index++] += xp * y0 * z0;
      dens[index] += xp * y0 * zp;
      index += grid->ngrid2() - 2;  // Step to the next row in y
      dens[index++] += xp * yp * zm;
      dens[index++] += xp * yp * z0;
      dens[index] += xp * yp * zp;
    }
  }

  /* ------------------------------------------------------------------- */

  Float count() { return count_; }

 private:
  int count_;  // The number of galaxies read in.
  // The sum of squares of the weights, which is the shot noise for P_0.
  Float Pshot_;
};

#endif  // READ_GALAXIES_H