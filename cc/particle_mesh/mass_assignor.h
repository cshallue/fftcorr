#ifndef MASS_ASSIGNMENT_H
#define MASS_ASSIGNMENT_H

#include <memory>
#include <vector>

// TODO: use include paths in the makefile compiler command
#include "../array/row_major_array.h"
#include "../grid/config_space_grid.h"
#include "../multithreading.h"
#include "../profiling/timer.h"
#include "../types.h"
#include "merge_sort_omp.h"
#include "particle.h"
#include "window_functions.h"

class MassAssignor {
 public:
  MassAssignor(ConfigSpaceGrid *grid, uint64 buffer_size)
      : grid_(grid),
        window_func_(make_window_function(grid->window_type())),
        buffer_size_(buffer_size),
        count_(0),
        skipped_(0),
        totw_(0),
        totwsq_(0) {
    // TODO: for development, remove.
#ifdef OPENMP
    fprintf(stderr, "# Running with %d threads\n", omp_get_max_threads());
#else
    fprintf(stderr, "# Running single threaded.\n");
#endif
    gal_.reserve(buffer_size_);
  }

  uint64 count() const { return count_; }
  uint64 skipped() const { return skipped_; }
  Float totw() const { return totw_; }
  Float totwsq() const { return totwsq_; }
  Float sort_time() const { return sort_time_.elapsed_sec(); }
  Float window_time() const { return window_time_.elapsed_sec(); }

  void clear() {
    gal_.clear();
    count_ = 0;
    skipped_ = 0;
    totw_ = 0;
    totwsq_ = 0;
    sort_time_.clear();
    window_time_.clear();
  }

  void add_particles_to_buffer(const RowMajorArrayPtr<Float, 2> &posw) {
    assert(posw.shape(1) == 4);
    const Float *row;
    for (int i = 0; i < posw.shape(0); ++i) {
      row = posw.get_row(i);
      add_particle_to_buffer(row[0], row[1], row[2], row[3]);
    }
  }

  void add_particles_to_buffer(const RowMajorArrayPtr<Float, 2> &pos,
                               const ArrayPtr1D<Float> weights) {
    assert(pos.shape(1) == 3);
    assert(pos.shape(0) == weights.shape(0));
    const Float *row;
    const Float *w = weights.data();
    for (int i = 0; i < pos.shape(0); ++i) {
      row = pos.get_row(i);
      add_particle_to_buffer(row[0], row[1], row[2], w[i]);
    }
  }

  void add_particles_to_buffer(const RowMajorArrayPtr<Float, 2> &pos,
                               Float weight) {
    assert(pos.shape(1) == 3);
    const Float *row;
    for (int i = 0; i < pos.shape(0); ++i) {
      row = pos.get_row(i);
      add_particle_to_buffer(row[0], row[1], row[2], weight);
    }
  }

  void add_particle_to_buffer(Float x, Float y, Float z, Float w) {
    grid_->change_survey_to_grid_coords(x, y, z);
    uint64 index = grid_->data().get_index(floor(x), floor(y), floor(z));
    if (index < 0 || index >= grid_->size()) {
      // Expected on rare occasions where a particle is within numerical
      // precision of a right boundary.
      // fprintf(stderr,
      //         "Skipping particle outside grid range: index = %llu, expected
      //         to be in range [0, %llu)\n", index, grid_->size());
      skipped_ += 1;
      return;
    }
    gal_.push_back(Particle(x, y, z, w, index));
    if (gal_.size() >= buffer_size_) {
      flush();
    }
    count_ += 1;
    totw_ += w;
    totwsq_ += w * w;
  }

  void flush() {
    // Given a set of Galaxies, add them to the grid and then reset the list
    const int galsize = gal_.size();

    // If we're parallelizing this, then we need to keep the threads
    // from stepping on each other.  Do this in slabs, but with only every third
    // slab active at any time.

    // Let's sort the particles by x.
    // Need to supply an equal amount of temporary space to merge sort.
    // Do this by another vector.
    buf_.reserve(galsize);
    sort_time_.start();
    mergesort_parallel_omp(gal_.data(), galsize, buf_.data(),
                           omp_get_max_threads());
    sort_time_.stop();
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
    window_time_.start();
    for (int mod = 0; mod < slabset; mod++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int x = mod; x < ngrid[0]; x += slabset) {
        // For each slab, insert these particles
        for (int j = first[x]; j < first[x + 1]; j++)
          window_func_->add_particle_to_grid(gal_[j], &grid_->data());
      }
    }
    window_time_.stop();

    gal_.clear();
  }

 private:
  ConfigSpaceGrid *grid_;
  std::unique_ptr<WindowFunction> window_func_;

  uint64 buffer_size_;
  std::vector<Particle> gal_;
  std::vector<Particle> buf_;  // Used for mergesort.

  uint64 count_;
  uint64 skipped_;
  Float totw_;
  Float totwsq_;

  mutable Timer sort_time_;
  mutable Timer window_time_;
};

#endif  // MASS_ASSIGNMENT_H
