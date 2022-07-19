#ifndef CATALOG_READER_H
#define CATALOG_READER_H

#include <assert.h>

#include <array>

#include "particle_mesh/mass_assignor.h"
#include "profiling/timer.h"
#include "types.h"

#define BUFFER_SIZE 512

class CatalogReader {
 public:
  void read_header(const char* filename, std::array<Float, 3>& posmin,
                   std::array<Float, 3>& posmax) {
    // Read posmin_[3], posmax_[3], max_sep_, blank8;
    FILE* fp = fopen(filename, "rb");
    assert(fp != NULL);
    double buf[8];
    int nread = fread(buf, sizeof(double), 8, fp);
    assert(nread == 8);
    fclose(fp);

    Float TOOBIG = 1e10;
    for (int i = 0; i < 7; i++) {
      assert(fabs(buf[i]) < TOOBIG);
    }
    posmin[0] = buf[0];
    posmin[1] = buf[1];
    posmin[2] = buf[2];
    posmax[0] = buf[3];
    posmax[1] = buf[4];
    posmax[2] = buf[5];
    // buf[6] and buf[7] not used, just for alignment.

    fprintf(stderr, "Reading survey box from header of %s\n", filename);
    fprintf(stderr, "posmin = [%.4f, %.4f, %.4f]\n", posmin[0], posmin[1],
            posmin[2]);
    fprintf(stderr, "posmax = [%.4f, %.4f, %.4f]\n", posmax[0], posmax[1],
            posmax[2]);
  }

  void read_galaxies(const char* filename, MassAssignor* mass_assignor) {
    total_time_.start();
    double tmp[8];
    double* b;

    int count = 0;
    fprintf(stdout, "# Reading from file named %s\n", filename);
    FILE* fp = fopen(filename, "rb");
    assert(fp != NULL);
    int nread = fread(tmp, sizeof(double), 8, fp);
    assert(nread == 8);  // Skip the header
    while ((nread = fread(&buffer_, sizeof(double), BUFFER_SIZE, fp)) > 0) {
      b = buffer_;
      grid_time_.start();
      for (int j = 0; j < nread; j += 4, b += 4) {
        mass_assignor->add_particle_to_buffer(b, b[3]);
        ++count;
      }
      grid_time_.stop();
      if (nread != BUFFER_SIZE) break;
    }
    fprintf(stdout, "# Found %d galaxies in this file\n", count);
    fclose(fp);
    // Add the remaining galaxies to the grid
    mass_assignor->flush();
    total_time_.stop();
  }

  Float total_time() const { return total_time_.elapsed_sec(); }
  Float grid_time() const { return grid_time_.elapsed_sec(); }
  Float io_time() const { return total_time() - grid_time(); }

 private:
  double buffer_[BUFFER_SIZE];
  Timer total_time_;
  Timer grid_time_;
};

#endif  // CATALOG_READER_H