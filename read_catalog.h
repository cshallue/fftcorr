#ifndef READ_GALAXIES_H
#define READ_GALAXIES_H

#include <assert.h>

#include "particle_mesh/src/mass_assignor.h"
#include "types.h"

#define BUFFER_SIZE 512

// TODO: rename this file, or class, or both. Possibly combine with SurveyBox
// again.
class SurveyReader {
 public:
  SurveyReader(MassAssignor *mass_assignor)
      : mass_assignor_(mass_assignor), count_(0), totw_(0), totwsq_(0) {}

  int count() { return count_; }
  Float totw() { return totw_; }
  Float totwsq() { return totwsq_; }

  // TODO: this function only needs to read one file at a time.
  void read_galaxies(const char filename[]) {
    // filename and filename2 are the input particles. filename2==NULL
    // will skip that one
    // Read to the end of the file, bringing in x,y,z,w points.
    // Bin them onto the grid.
    // We're setting up a large buffer to read in the galaxies.
    // Will reset the buffer periodically, just to limit the size.
    double tmp[8];
    double *b;

    // IO.Start();
    int thiscount = 0;
    fprintf(stdout, "# Reading from file named %s\n", filename);
    FILE *fp = fopen(filename, "rb");
    assert(fp != NULL);
    int nread = fread(tmp, sizeof(double), 8, fp);
    assert(nread == 8);  // Skip the header
    while ((nread = fread(&buffer_, sizeof(double), BUFFER_SIZE, fp)) > 0) {
      b = buffer_;
      for (int j = 0; j < nread; j += 4, b += 4) {
        mass_assignor_->add_galaxy(b);
        ++thiscount;
        totw_ += b[3];
        totwsq_ += b[3] * b[3];
      }
      if (nread != BUFFER_SIZE) break;
    }
    count_ += thiscount;
    fprintf(stdout, "# Found %d galaxies in this file\n", thiscount);
    fclose(fp);
    // IO.Stop();
    // Add the remaining galaxies to the grid
    // mass_assignor_->flush_to_density_field();
  }

 private:
  MassAssignor *mass_assignor_;

  double buffer_[BUFFER_SIZE];

  int count_;  // The number of galaxies read in.

  Float totw_;
  // The sum of squares of the weights, which is the shot noise for P_0.
  Float totwsq_;
};

#endif  // READ_GALAXIES_H