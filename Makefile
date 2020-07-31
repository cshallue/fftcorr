# For my MacBook
FFTW = -L ~/miniconda3/envs/fftcorr/lib -I ~/miniconda3/envs/fftcorr/include -lfftw3
CXXFLAGS = -Wall -O2 
#CXXFLAGS = -g ${FFTW} 

# For Odyssey
#FFTW = -L ~/fftw-3.3.5/lib/ -I ~/fftw-3.3.5/include/ -lfftw3

OMP = -DOPENMP -DFFTSLAB -DSLAB

# If you want to run with multi-threading, uncomment the following two lines
#CXX = g++ -march=native -fopenmp -lgomp 
#-fopt-info-vec-missed -fopt-info-vec-optimized
#CXXFLAGS = -O3 ${OMP}

# Or is you want multi-threading with icc, the following would work:
# ICC not tested!  And one may need to compile FFTW with it.
#CXX = icc -liomp5 -openmp
#CXXFLAGS = -O2 -Wall ${OMP}

default: fftcorr

fftcorr: fftcorr.cpp Makefile merge_sort_omp.cpp STimer.cc
	${CXX} ${CXXFLAGS} fftcorr.cpp ${FFTW} -o fftcorr

tar:
	tar cvf fftcorr.tar --exclude="*.pyc" \
	    Makefile STimer.cc fftcorr.cpp fftcorr.py wcdm merge_sort_omp.cpp d12.cpp generate_command.py
