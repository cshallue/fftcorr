#ifndef READ_GALAXIES_H
#define READ_GALAXIES_H

#include <assert.h>

#include "particle_mesh/mass_assignor.h"
#include "profiling/timer.h"
#include "types.h"

#define BUFFER_SIZE 512

// TODO: rename this file, or class, or both. Possibly combine with SurveyBox
// again.
class SurveyReader {
 public:
  SurveyReader(MassAssignor *mass_assignor) : mass_assignor_(mass_assignor) {}

  void read_galaxies(const char filename[]) {
    total_time_.start();
    double tmp[8];
    double *b;

    int count = 0;
    fprintf(stdout, "# Reading from file named %s\n", filename);
    FILE *fp = fopen(filename, "rb");
    assert(fp != NULL);
    int nread = fread(tmp, sizeof(double), 8, fp);
    assert(nread == 8);  // Skip the header
    while ((nread = fread(&buffer_, sizeof(double), BUFFER_SIZE, fp)) > 0) {
      b = buffer_;
      for (int j = 0; j < nread; j += 4, b += 4) {
        mass_assignor_->add_particle_to_buffer(b[0], b[1], b[2], b[3]);
        ++count;
      }
      if (nread != BUFFER_SIZE) break;
    }
    fprintf(stdout, "# Found %d galaxies in this file\n", count);
    fclose(fp);
    // Add the remaining galaxies to the grid
    // mass_assignor_->flush();
    total_time_.stop();
  }

  Float total_time() const { return total_time_.elapsed_sec(); }

 private:
  MassAssignor *mass_assignor_;

  double buffer_[BUFFER_SIZE];

  Timer total_time_;
};

#endif  // READ_GALAXIES_H