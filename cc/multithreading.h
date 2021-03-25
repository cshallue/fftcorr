#ifndef MULTITHREADING_H
#define MULTITHREADING_H

// #define SLAB		// Handle the arrays by associating threads to x-slabs
// #define FFTSLAB  	// Do the FFTs in 2D, then 1D
// #define OPENMP	// Turn on the OPENMP items

#ifndef OPENMP
// We probably don't want to do this if single threaded.
#undef SLAB
#endif  // OPENMP


#ifdef OPENMP
#include <omp.h>
#else
// Fake these for the single-threaded case to make code easier to read.
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif  // OPENMP


#ifdef SLAB
// Treating the matrices explicitly by x slab might allow NUMA memory to be
// closer to the socket running the thread.
#define MY_SCHEDULE schedule(static, 1)
#define YLM_SCHEDULE schedule(static, 1)
#else
// Just treat the matrices as one big object.
#define MY_SCHEDULE schedule(dynamic, 512)
#define YLM_SCHEDULE schedule(dynamic, 1)
#endif  // SLAB

#endif  // MULTITHREADING_H
