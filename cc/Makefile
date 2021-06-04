# For my MacBook
FFTW_INCLUDE = -I ~/miniconda3/envs/fftcorr/include
FFTW = ${FFTW_INCLUDE} -L ~/miniconda3/envs/fftcorr/lib -lfftw3
CXXFLAGS = -std=c++0x -Wall
#CXXFLAGS = -g ${FFTW} 

# For Odyssey
#FFTW = -L ~/fftw-3.3.5/lib/ -I ~/fftw-3.3.5/include/ -lfftw3

use_omp = false
ifeq ($(use_omp),true)
  # http://www.fftw.org/fftw3_doc/Usage-of-Multi_002dthreaded-FFTW.html
  FFTW += -lfftw3_threads -lm
  CXX = g++ -march=native -fopenmp -lgomp # -fopt-info-vec-missed -fopt-info-vec-optimized
  CXXFLAGS += -O3 -DOPENMP # -DFFTSLAB -DSLAB
else
  CXXFLAGS += -O2 -Wno-unknown-pragmas # -DFFTSLAB -DSLAB
endif

# If you want multi-threading with icc, the following would work:
# ICC not tested!  And one may need to compile FFTW with it.
#CXX = icc -liomp5 -openmp
#CXXFLAGS = -O2 -Wall ${OMP}

default: fftcorr

fftcorr: fftcorr.cc fft_grid.o array_ops.o timer.o
	${CXX} ${CXXFLAGS} fftcorr.cc -o fftcorr fft_grid.o array_ops.o timer.o ${FFTW}

array_ops.o: array/array_ops.cc array/array_ops.h types.h
	${CXX} ${CXXFLAGS} -c array/array_ops.cc

fft_grid.o: grid/fft_grid.cc grid/fft_grid.h types.h array_ops.o timer.o
	${CXX} ${CXXFLAGS} -c grid/fft_grid.cc ${FFTW_INCLUDE}

timer.o: profiling/timer.cc profiling/timer.h
	${CXX} ${CXXFLAGS} -c profiling/timer.cc

clean:
	rm fftcorr array_ops.o fft_grid.o

tar:
	tar cvf fftcorr.tar --exclude="*.pyc" \
	    Makefile STimer.cc fftcorr.cpp fftcorr.py wcdm merge_sort_omp.cpp d12.cpp generate_command.py
