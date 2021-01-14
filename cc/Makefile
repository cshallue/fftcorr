# For my MacBook
FFTW_INCLUDE = -I ~/miniconda3/envs/fftcorr/include
FFTW = ${FFTW_INCLUDE} -L ~/miniconda3/envs/fftcorr/lib -lfftw3
CXXFLAGS = -std=c++0x -Wall -O2 
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

fftcorr: fftcorr.cc array3d.o discrete_field.o
	${CXX} ${CXXFLAGS} fftcorr.cc -o fftcorr array3d.o discrete_field.o ${FFTW}

array3d.o: array3d.cc array3d.h types.h
	${CXX} ${CXXFLAGS} -c array3d.cc

discrete_field.o: discrete_field.cc discrete_field.h array3d.h types.h array3d.o
	${CXX} ${CXXFLAGS} ${FFTW_INCLUDE} -c discrete_field.cc

clean:
	rm fftcorr array3d.o

tar:
	tar cvf fftcorr.tar --exclude="*.pyc" \
	    Makefile STimer.cc fftcorr.cpp fftcorr.py wcdm merge_sort_omp.cpp d12.cpp generate_command.py
