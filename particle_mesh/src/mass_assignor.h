#ifndef MASS_ASSIGNMENT_H
#define MASS_ASSIGNMENT_H

#include <memory>
#include <vector>

// TODO: use include paths in the makefile compiler command
#include "../../galaxy.h"
#include "../../grid/config_space_grid.h"
#include "../../multithreading.h"
#include "../../types.h"
#include "merge_sort_omp.h"
#include "window_functions.h"

// TODO: transition galaxy -> particle everywhere
class MassAssignor {
 public:
  MassAssignor(ConfigSpaceGrid *grid, WindowType window_type, int buffer_size)
      : grid_(grid),
        window_func_(make_window_function(window_type)),
        buffer_size_(buffer_size),
        count_(0),
        totw_(0),
        totwsq_(0) {
    gal_.reserve(buffer_size_);
  }

  int count() { return count_; }
  Float totw() { return totw_; }
  Float totwsq() { return totwsq_; }

  // TODO: this could be vectorized when posw is a matrix.
  void add_particle(Float posw[4]) {
    grid_->change_survey_to_grid_coords(posw);
    uint64 index =
        grid_->data().get_index(floor(posw[0]), floor(posw[1]), floor(posw[2]));
    gal_.push_back(Galaxy(posw, index));
    if (gal_.size() >= buffer_size_) {
      // IO.Stop();
      flush();
      // IO.Start();
    }
    count_ += 1;
    totw_ += posw[3];
    totwsq_ += posw[3] * posw[3];
  }

  void flush() {
    // Given a set of Galaxies, add them to the grid and then reset the list
    // CIC.Start();
    const int galsize = gal_.size();

    // If we're parallelizing this, then we need to keep the threads
    // from stepping on each other.  Do this in slabs, but with only every third
    // slab active at any time.

    // Let's sort the particles by x.
    // Need to supply an equal amount of temporary space to merge sort.
    // Do this by another vector.
    buf_.reserve(galsize);
    mergesort_parallel_omp(gal_.data(), galsize, buf_.data(),
                           omp_get_max_threads());
    // This just falls back to std::sort if omp_get_max_threads==1

    // Now we need to find the starting point of each slab
    // Galaxies between N and N+1 should be in indices [first[N], first[N+1]).
    // That means that first[N] should be the index of the first particle to
    // exceed N.
    const std::array<int, 3> &ngrid = grid_->ngrid();
    int first[ngrid[0] + 1];
    int ptr = 0;
    for (int j = 0; j < galsize; j++) {
      while (gal_[j].x > ptr) first[ptr++] = j;
    }
    for (; ptr <= ngrid[0]; ptr++) first[ptr] = galsize;

    // Now, we'll loop, with each thread in charge of slab x.
    // Not bothering with NUMA issues.  a) Most of the time is spent waiting
    // for memory to respond, not actually piping between processors.  b)
    // Adjacent slabs may not be on the same memory bank anyways.  Keep it
    // simple.
    int slabset = window_func_->width();
    for (int mod = 0; mod < slabset; mod++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int x = mod; x < ngrid[0]; x += slabset) {
        // For each slab, insert these particles
        for (int j = first[x]; j < first[x + 1]; j++)
          window_func_->add_particle_to_grid(gal_[j], &grid_->data().arr());
      }
    }

    gal_.clear();
    // CIC.Stop();
  }

 private:
  ConfigSpaceGrid *grid_;
  std::unique_ptr<WindowFunction> window_func_;

  int buffer_size_;
  std::vector<Galaxy> gal_;
  std::vector<Galaxy> buf_;  // Used for mergesort.

  int count_;
  Float totw_;
  Float totwsq_;
};

#endif  // MASS_ASSIGNMENT_H