#ifndef MASS_ASSIGNMENT_H
#define MASS_ASSIGNMENT_H

#include <array>
#include <memory>
#include <vector>

// TODO: use include paths in the makefile compiler command
#include "../array/row_major_array.h"
#include "../grid/config_space_grid.h"
#include "../multithreading.h"
#include "../profiling/timer.h"
#include "../types.h"
#include "merge_sort_omp.h"
#include "window_functions.h"

struct Particle {
  Particle(const Float *_pos, Float _w, uint64 _index)
      : pos({_pos[0], _pos[1], _pos[2]}), w(_w), index(_index) {}

  // So we can sort in index order.
  bool operator<(const Particle &other) const { return index < other.index; }

  std::array<Float, 3> pos;
  Float w;
  uint64 index;
};

class MassAssignor {
 public:
  MassAssignor(ConfigSpaceGrid &grid, bool periodic_wrap, uint64 buffer_size)
      : grid_(grid),
        window_func_(make_window_function(grid.window_type())),
        periodic_wrap_(periodic_wrap),
        buffer_size_(buffer_size),
        num_added_(0),
        num_skipped_(0),
        totw_(0),
        totwsq_(0) {
    gal_.reserve(buffer_size_);
#ifdef OPENMP
    fprintf(stderr, "# Running with %d threads\n", omp_get_max_threads());
#else
    // TODO: for development, remove.
    fprintf(stderr, "# Running single threaded with buffer size %lld.\n",
            buffer_size_);
#endif  // OPENMP
  }

  int buffer_size() const { return buffer_size_; }
  uint64 num_added() const { return num_added_; }
  uint64 num_skipped() const { return num_skipped_; }
  Float totw() const { return totw_; }
  Float totwsq() const { return totwsq_; }
  Float sort_time() const { return sort_time_.elapsed_sec(); }
  Float window_time() const { return window_time_.elapsed_sec(); }

  void clear() {
#ifdef OPENMP
    gal_.clear();
#endif  // OPENMP
    num_added_ = 0;
    num_skipped_ = 0;
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
      add_particle_to_buffer(row, row[3]);
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
      add_particle_to_buffer(row, w[i]);
    }
  }

  void add_particles_to_buffer(const RowMajorArrayPtr<Float, 2> &pos,
                               Float weight) {
    assert(pos.shape(1) == 3);
    const Float *row;
    for (int i = 0; i < pos.shape(0); ++i) {
      row = pos.get_row(i);
      add_particle_to_buffer(row, weight);
    }
  }

  void add_particle_to_buffer(const Float *pos, Float w) {
    Float x[3];  // Grid coordinates of particle.
    if (!grid_.get_grid_coords(pos, periodic_wrap_, x)) {
      // Particle is outside the grid boundary and we're not periodic wrapping.
      num_skipped_ += 1;
      return;
    }
    if (!buffer_size_) {
      // We're adding particles directly to the grid.
      window_time_.start();
      add_particle_to_grid(x, w);
      window_time_.stop();
    } else {
      uint64 i = grid_.data().get_index(floor(x[0]), floor(x[1]), floor(x[2]));
      gal_.emplace_back(x, w, i);
      if (gal_.size() >= buffer_size_) flush();
    }
    num_added_ += 1;
    totw_ += w;
    totwsq_ += w * w;
  }

  void flush() {
    if (!buffer_size_) return;

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
#ifdef OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif  // OPENMP
    mergesort_parallel_omp(gal_.data(), galsize, buf_.data(), max_threads);
    sort_time_.stop();
    // This just falls back to std::sort if omp_get_max_threads==1

    // Now we need to find the starting point of each slab
    // Galaxies between N and N+1 should be in indices [first[N], first[N+1]).
    // That means that first[N] should be the index of the first particle to
    // exceed N.
    const std::array<int, 3> &ngrid = grid_.shape();
    int first[ngrid[0] + 1];
    int ptr = 0;
    for (int j = 0; j < galsize; j++) {
      while (gal_[j].pos[0] >= ptr) first[ptr++] = j;
    }
    for (; ptr <= ngrid[0]; ptr++) first[ptr] = galsize;

    // Now, we'll loop, with each thread in charge of slab x.
    // Not bothering with NUMA issues.  a) Most of the time is spent waiting
    // for memory to respond, not actually piping between processors.  b)
    // Adjacent slabs may not be on the same memory bank anyways.  Keep it
    // simple.
    int width = window_func_->width();
    // We shouldn't overlap the start of the grid with the end of the grid, due
    // to periodic wrapping. The maximum end slab index that will not overlap
    // with a given start index is start + maxsep.
    int maxsep = width * (ngrid[0] / width - 1);
    window_time_.start();
    for (int start = 0; start < width; ++start) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int x = start; x <= start + maxsep; x += width) {
        for (int j = first[x]; j < first[x + 1]; ++j) {
          add_particle_to_grid(gal_[j]);
        }
      }
    }
    // Slabs at the end of the grid that were cut off to prevent overlap.
    for (int x = width + maxsep; x < ngrid[0]; ++x) {
      for (int j = first[x]; j < first[x + 1]; ++j) {
        add_particle_to_grid(gal_[j]);
      }
    }
    window_time_.stop();

    gal_.clear();
  }

 private:
  void add_particle_to_grid(const Float *pos, Float weight) {
    window_func_->add_particle_to_grid(pos, weight, grid_.data());
  }

  void add_particle_to_grid(const Particle &p) {
    add_particle_to_grid(p.pos.data(), p.w);
  }

  // bool apply_displacement_to_grid_coord(int i, Float &x,
  //                                       const Float *dxyz) const {
  //   x += dxyz[i] / grid_.cell_size();
  //   return maybe_wrap_coord(i, x);
  // }

  // bool apply_displacement_to_grid_coords(Float &x, Float &y, Float &z) const
  // {
  //   if (!disp_) return true;  // Not applying any displacement.
  //   const Float *dxyz = disp_->get_row(floor(x), floor(y), floor(z));
  //   return apply_displacement_to_grid_coord(0, x, dxyz) &&
  //          apply_displacement_to_grid_coord(1, y, dxyz) &&
  //          apply_displacement_to_grid_coord(2, z, dxyz);
  // }

  ConfigSpaceGrid &grid_;
  std::unique_ptr<WindowFunction> window_func_;

  // Whether the grid should be treated as periodic.
  const bool periodic_wrap_;

  const uint64 buffer_size_;
  std::vector<Particle> gal_;  // TODO: name -> particles
  std::vector<Particle> buf_;  // Used for mergesort.

  uint64 num_added_;  // TODO: rename to something more descriptive
  uint64 num_skipped_;
  Float totw_;
  Float totwsq_;

  mutable Timer sort_time_;
  mutable Timer window_time_;
};

#endif  // MASS_ASSIGNMENT_H
