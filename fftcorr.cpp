/* fftcorr.cpp -- Daniel Eisenstein, July 2016
This computes anisotropic correlation functions using the FFT techniques of 
Slepian & Eisenstein 2015.

Input particles files have a specific binary format:

64 bytes header:  double posmin[3], posmax[3], max_sep, blank8;
Then 4 doubles per particle: x,y,z,weight, repeating for each particle.

Posmin and posmax are the bounding box for the particle data.
Input posmin and posmax must be oversized so that no periodic replicas are
within max_sep of each other.  Don't forget to include the cell-size
and CIC effects in this estimate; we recommend padding by another
50 Mpc or so, just to be safe.  This code does not periodic wrap, so 
it is required that posmin and posmax be padded. 

There is a python code that can check the cubic box case without
CIC.  The last time I checked, it matched to 9 digits, so one can
do very well.  Remember that this is only true for NEAREST_CELL.

I have not compared in detail to a pair-counting code, but I did
verify that the monopole of RR matches to O(10%).  That is, the
basic normalization makes sense.  Also, results are stable as the
grid changes; no grid normalization issues.  Padding the box out
further (with constant cell size and registration) doesn't change
the results at all, as one would expect.


Memory usage: Dominated by 3 double-precision arrays of size Nx*Ny*Nz.
Note that the 3 box dimensions need not be equal, although one should pick 
FFT-friendly values.  There is value in rotating the survey to fit in a
minimum rectangular box.  

Use the -cell (-c) option to force a cell size, which may then
oversize the box.  This might be useful in combining disjoint survey
regions, for example.

Use the -periodic (-p) option to configure for a cubic box.  In 
this case, the posmin/posmax inputs should not have any padding,
but should reflect the periodic wrapping length.  The observer
will be placed far away in the -x direction from the center of
the box.

Using the -zeromean (-z) option will invoke -periodic and then also
set the mean density of the box to zero.  This may allow one to
avoid using a random catalog for zero mean cases, as it is the
equivalent to entering N=D-R with an infinite set of R, but this
requires further testing.

Run time: On a single-core MacBook Pro, 512^3, ell=2 is taking about
40 seconds.  The time is 90+% dominated by the FFTs.  Computing the
Ylm's is 5%.  Both of these scale with Nx*Ny*Nz.

The primary correlation routine is multi-threaded.  On Odyssey
(64-core Athlons), it takes about 45 seconds to do 1024*1024*768
to ell=4.  

On the 4-socket Opteron machine, I found noticeable performance gains if
the 3 main arrays were assigned YZ slabs to individual (and consistent)
threads.  This is because the memory then ends up on the memory bank
specific to each thread.  Only the X FFT needs to cross memory banks.
Use -DSLAB -DFFTSLAB to turn this on.  I peak around 50-60 GFLOPS
for this case.

Loading and gridding the particles is subdominant for BOSS-sized
problems.  But this would be much larger if we had a particle density
typical of N-body simulations rather than SDSS BOSS galaxy density!
This part is multi-threaded and does sorting to try to help the 
cloud-in-cell to keep up.  

*/

/* ======================= Compile-time user flags ================= */
// #define NEAREST_CELL   // To turn-off triangular CIC, e.g., for comparing to python
#define WAVELET   	// To turn on a D12 wavelet density assignment
// #define SLAB		// Handle the arrays by associating threads to x-slabs
// #define FFTSLAB  	// Do the FFTs in 2D, then 1D
// #define OPENMP	// Turn on the OPENMP items

/* ======================= Preamble ================= */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <fftw3.h>
#include "STimer.cc"

// For multi-threading:
#ifdef OPENMP
    #include <omp.h>
#else
    // Just fake these for the single-threaded case, to make code easier to read.
    int omp_get_max_threads() { return 1; }
    int omp_get_num_threads() { return 1; }
    int omp_get_thread_num() { return 0; }
#endif

#define PAGE 4096     // To force some memory alignment; unit is bytes.

// In principle, this gives flexibility, but one also would need to adjust the FFTW calls.
typedef double Float;    
typedef std::complex<double> Complex;

// We want to allow that the FFT grid could exceed 2^31 cells
typedef unsigned long long int uint64;

STimer IO, Setup, FFTW, Correlate, YlmTime, Total, CIC, Misc,
       FFTonly, Hist, Extract, AtimesB, Init, FFTyz, FFTx;


#include "d12.cpp"

/* ========================================================================= */

class Histogram {
  // This should set up the binning and the space to hold the answers
  public:
    int maxell;
    Float sep;
    int nbins;

    Float *cnt;
    Float *hist;
    Float binsize;
    Float zerolag;    // The value at zero lag

    Histogram(int _maxell, Float _sep, Float _dsep) {
	int err;
	maxell = _maxell;
	sep = _sep;
	binsize = _dsep;
	zerolag = -12345.0;
	nbins = floor(sep/binsize);
	assert(nbins>0&&nbins<1e6);  // Too big probably means data entry error.

	// Allocate cnt[nbins], hist[maxell/2+1, nbins]
	err=posix_memalign((void **) &cnt, PAGE, sizeof(Float)*nbins); assert(err==0);
	err=posix_memalign((void **) &hist, PAGE, sizeof(Float)*nbins*(maxell/2+1)); assert(err==0);
	assert(cnt!=NULL);
	assert(hist!=NULL);
    }
    ~Histogram() {
	// For some reason, these cause a crash!  Weird!
        // free(hist);
        // free(cnt);
    }

    // TODO: Might consider creating more flexible ways to select a binning.
    inline int r2bin(Float r) {
        return floor(r/binsize);
    }

    void histcorr(int ell, int n, Float *rnorm, Float *total) {
        // Histogram into bins by rnorm[n], adding up weighting by total[n].
	// Add to multipole ell.
	if (ell==0) {
	    for (int j=0; j<nbins; j++) cnt[j] = 0.0;
	    for (int j=0; j<nbins; j++) hist[j] = 0.0;
	    for (int j=0; j<n; j++) {
		int b = r2bin(rnorm[j]);
		if (rnorm[j]<binsize*1e-6) {
		    zerolag = total[j];
		}
		if (b>=nbins||b<0) continue;
		cnt[b]++;
		hist[b]+= total[j];
	    }
	} else {
	    // ell>0
	    Float *h = hist+ell/2*nbins;
	    for (int j=0; j<nbins; j++) h[j] = 0.0;
	    for (int j=0; j<n; j++) {
		int b = r2bin(rnorm[j]);
		if (b>=nbins||b<0) continue;
		h[b]+= total[j];
	    }
	}
    }

    Float sum() {
         // Add up the histogram values for ell=0
	 Float total=0.0;
	 for (int j=0; j<nbins; j++) total += hist[j];
	 return total;
    }

    void print(FILE *fp, int norm) {
        // Print out the results
	// If norm==1, divide by counts
	Float denom;
	for (int j=0; j<nbins; j++) {
	    fprintf(fp,"%1d ", norm);
	    if (sep>2)
		fprintf(fp,"%6.2f %8.0f", (j+0.5)*binsize, cnt[j]);
	    else
		fprintf(fp,"%7.4f %8.0f", (j+0.5)*binsize, cnt[j]);
	    if (cnt[j]!=0&&norm) denom = cnt[j]; else denom = 1.0;
	    for (int k=0; k<=maxell/2; k++) 
		fprintf(fp," %16.9e", hist[k*nbins+j]/denom);
	    fprintf(fp,"\n");
	}
    }
};    // end Histogram

/* ======================================================================= */
// Here are some matrix handling routines, which may need OMP attention


#ifndef OPENMP
    #undef SLAB		// We probably don't want to do this if single threaded
#endif

#ifdef SLAB
    // Try treating the matrices explicitly by x slab;
    // this might allow NUMA memory to be closer to the socket running the thread.
    #define MY_SCHEDULE schedule(static,1)
    #define YLM_SCHEDULE schedule(static,1)
#else
    // Just treat the matrices as one big object
    #define MY_SCHEDULE schedule(dynamic,512)
    #define YLM_SCHEDULE schedule(dynamic,1)
#endif

void initialize_matrix(Float *&m, const uint64 size, const int nx) {
    // Initialize a matrix m and set it to zero.
    // We want to touch the whole matrix, because in NUMA this defines the association
    // of logical memory into the physical banks.
    // nx will be our slab decomposition; it must divide into size evenly
    // Warning: This will only allocate new space if m==NULL.  This allows
    // one to reuse space.  But(!!) there is no check that you've not changed
    // the size of the matrix -- you could overflow the previously allocated
    // space.
    assert (size%nx==0);
    Init.Start();
    if (m==NULL) {
	int err=posix_memalign((void **) &m, PAGE, sizeof(Float)*size+PAGE); assert(err==0);
    }
    assert(m!=NULL);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
	Float *mslab = m+x*nyz;
	for (uint64 j=0; j<nyz; j++) mslab[j] = 0.0;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) m[j] = 0.0;
#endif
    Init.Stop();
    return;
}

void initialize_matrix_by_copy(Float *&m, const uint64 size, const int nx, Float *copy) {
    // Initialize a matrix m and set it to copy[size].
    // nx will be our slab decomposition; it must divide into size evenly
    // Warning: This will only allocate new space if m==NULL.  This allows
    // one to reuse space.  But(!!) there is no check that you've not changed
    // the size of the matrix -- you could overflow the previously allocated
    // space.
    assert (size%nx==0);
    Init.Start();
    if (m==NULL) {
	int err=posix_memalign((void **) &m, PAGE, sizeof(Float)*size+PAGE); assert(err==0);
    }
    assert(m!=NULL);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
	Float *mslab = m+x*nyz;
	Float *cslab = copy+x*nyz;
	for (uint64 j=0; j<nyz; j++) mslab[j] = cslab[j];
	// memcpy(mslab, cslab, sizeof(Float)*nyz);    // Slower for some reason!
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) m[j] = copy[j];
#endif
    Init.Stop();
    return;
}

void set_matrix(Float *a, const Float b, const uint64 size, const int nx) {
    // Set a equal to a scalar b
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
	Float *aslab = a+x*nyz;
	for (uint64 j=0; j<nyz; j++) aslab[j] = b;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] = b;
#endif
}

void scale_matrix(Float *a, const Float b, const uint64 size, const int nx) {
    // Multiply a by a scalar b
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
	Float *aslab = a+x*nyz;
	for (uint64 j=0; j<nyz; j++) aslab[j] *= b;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] *= b;
#endif
}

void addscalarto_matrix(Float *a, const Float b, const uint64 size, const int nx) {
    // Add scalar b to matrix a
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
	Float *aslab = a+x*nyz;
	for (uint64 j=0; j<nyz; j++) aslab[j] += b;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] += b;
#endif
}

void copy_matrix(Float *a, Float *b, const uint64 size, const int nx) {
    // Set a equal to a vector b
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
	Float *aslab = a+x*nyz;
	Float *bslab = b+x*nyz;
	for (uint64 j=0; j<nyz; j++) aslab[j] = bslab[j];
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] = b[j];
#endif
}

void copy_matrix(Float *a, Float *b, const Float c, const uint64 size, const int nx) {
    // Set a equal to a vector b times a scalar c
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
	Float *aslab = a+x*nyz;
	Float *bslab = b+x*nyz;
	for (uint64 j=0; j<nyz; j++) aslab[j] = bslab[j]*c;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] = b[j]*c;
#endif
}

Float sum_matrix(Float *a, const uint64 size, const int nx) {
    // Sum the elements of the matrix
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
    Float tot=0.0;
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (int x=0; x<nx; x++) {
	Float *aslab = a+x*nyz;
	for (uint64 j=0; j<nyz; j++) tot += aslab[j];
    }
#else
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (uint64 j=0; j<size; j++) tot += a[j];
#endif
    return tot;
}

Float sumsq_matrix(Float *a, const uint64 size, const int nx) {
    // Sum the square of elements of the matrix
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
    Float tot=0.0;
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (int x=0; x<nx; x++) {
	Float *aslab = a+x*nyz;
	for (uint64 j=0; j<nyz; j++) tot += aslab[j]*aslab[j];
    }
#else
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (uint64 j=0; j<size; j++) tot += a[j]*a[j];
#endif
    return tot;
}

void multiply_matrix_with_conjugation(Complex *a, Complex *b, const uint64 size, const int nx) {
    // Element-wise multiply a[] by conjugate of b[]
    // Note that size refers to the Complex size; the calling routine
    // is responsible for dividing the Float size by 2.
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
	Complex *aslab = a+x*nyz;
	Complex *bslab = b+x*nyz;
	for (uint64 j=0; j<nyz; j++) aslab[j] *= std::conj(bslab[j]);
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] *= std::conj(b[j]);
#endif
}

/* ==========================  Submatrix extraction =================== */

void extract_submatrix(Float *total, Float *corr, int csize[3], 
		       Float *work, int ngrid[3], const int ngrid2) {
    // Given a large matrix work[ngrid^3], 
    // extract out a submatrix of size csize^3, centered on work[0,0,0].
    // Multiply the result by corr[csize^3] and add it onto total[csize^3]
    // Again, zero lag is mapping to corr(csize/2, csize/2, csize/2),
    // but it is at (0,0,0) in the FFT grid.
    Extract.Start();
    int cx = csize[0]/2;    // This is the middle of the submatrix
    int cy = csize[1]/2;    // This is the middle of the submatrix
    int cz = csize[2]/2;    // This is the middle of the submatrix
    #pragma omp parallel for schedule(dynamic,1)
    for (uint64 i=0; i<csize[0]; i++) {
	uint64 ii = (ngrid[0]-cx+i)%ngrid[0];
	for (int j=0; j<csize[1]; j++) {
	    uint64 jj = (ngrid[1]-cy+j)%ngrid[1];
	    Float *t = total+(i*csize[1]+j)*csize[2];  // This is (i,j,0)
	    Float *cc = corr+(i*csize[1]+j)*csize[2];  // This is (i,j,0)
	    Float *Y =  work+(ii*ngrid[1]+jj)*ngrid2+ngrid[2]-cz;  
					// This is (ii,jj,ngrid[2]-c)
	    for (int k=0; k<cz; k++) t[k] += cc[k]*Y[k];
	    Y =  work+(ii*ngrid[1]+jj)*ngrid2-cz;  
					// This is (ii,jj,-c)
	    for (int k=cz; k<csize[2]; k++) t[k] += cc[k]*Y[k];
	}
    }
    Extract.Stop();
}

void extract_submatrix_C2R(Float *total, Float *corr, int csize[3], 
		       Complex *work, int ngrid[3], const int ngrid2) {
    // Given a large matrix work[ngrid^3/2], 
    // extract out a submatrix of size csize^3, centered on work[0,0,0].
    // The input matrix is Complex * with the half-domain Fourier convention.
    // We are only summing the real part; the imaginary part always sums to zero.
    // Need to reflect the -z part around the origin, which also means reflecting x & y.
    // ngrid[2] and ngrid2 are given as their Float values, not yet divided by two.
    // Multiply the result by corr[csize^3] and add it onto total[csize^3]
    // Again, zero lag is mapping to corr(csize/2, csize/2, csize/2),
    // but it is at (0,0,0) in the FFT grid.
    Extract.Start();
    int cx = csize[0]/2;    // This is the middle of the submatrix
    int cy = csize[1]/2;    // This is the middle of the submatrix
    int cz = csize[2]/2;    // This is the middle of the submatrix
    #pragma omp parallel for schedule(dynamic,1)
    for (uint64 i=0; i<csize[0]; i++) {
	uint64 ii = (ngrid[0]-cx+i)%ngrid[0];
	uint64 iin = (ngrid[0]-ii)%ngrid[0];   // The reflected coord
	for (int j=0; j<csize[1]; j++) {
	    uint64 jj = (ngrid[1]-cy+j)%ngrid[1];
	    uint64 jjn = (ngrid[1]-jj)%ngrid[1];   // The reflected coord
	    Float *t = total+(i*csize[1]+j)*csize[2];  // This is (i,j,0)
	    Float *cc = corr+(i*csize[1]+j)*csize[2];  // This is (i,j,0)
	    // The positive half-plane (inclusize)
	    Complex *Y =  work+(ii*ngrid[1]+jj)*ngrid2/2-cz;  
					// This is (ii,jj,-cz)
	    for (int k=cz; k<csize[2]; k++) t[k] += cc[k]*std::real(Y[k]);
	    // The negative half-plane (inclusize), reflected.  
	    // k=cz-1 should be +1, k=0 should be +cz
	    Y =  work+(iin*ngrid[1]+jjn)*ngrid2/2+cz;
					// This is (iin,jjn,+cz)
	    for (int k=0; k<cz; k++) t[k] += cc[k]*std::real(Y[-k]);
	}
    }
    Extract.Stop();
}

/* ===============================  FFTW wrapper routines ===================== */

void setup_FFTW(fftw_plan &fft,  fftw_plan &fftYZ,  fftw_plan &fftX, 
                fftw_plan &ifft, fftw_plan &ifftYZ, fftw_plan &ifftX,
	        int ngrid[3], const int ngrid2, Float *work) {
    // Setup the FFTW plans, possibly from disk, and save the wisdom
    fprintf(stdout,"# Planning the FFTs..."); fflush(NULL);
    FFTW.Start();
    FILE *fp = NULL;
    #ifdef OPENMP
	#ifndef FFTSLAB
	    { int errval = fftw_init_threads(); assert(errval); }
	    fftw_plan_with_nthreads(omp_get_max_threads());
	#endif
	#define WISDOMFILE "wisdom_fftw_omp"
    #else
	#define WISDOMFILE "wisdom_fftw"
    #endif
    #ifdef FFTSLAB
	#undef WISDOMFILE
	#define WISDOMFILE "wisdom_fftw"
    #endif
    fp = fopen(WISDOMFILE, "r");
    if (fp!=NULL) {
	fprintf(stdout,"Reading %s...", WISDOMFILE); fflush(NULL);
	fftw_import_wisdom_from_file(fp);
	fclose(fp);
    }

    #ifndef FFTSLAB
	// The following interface should work even if ngrid2 was 'non-minimal',
	// as might be desired by padding.
	int nfft[3], nfftc[3];
	nfft[0] = nfftc[0] = ngrid[0];
	nfft[1] = nfftc[1] = ngrid[1];
	nfft[2] = ngrid2;   // Since ngrid2 is always even, this will trick 
	nfftc[2]= nfft[2]/2;
		// FFTW to assume ngrid2/2 Complex numbers in the result, while
		// fulfilling that nfft[2]>=ngrid[2].
	fft = fftw_plan_many_dft_r2c(3, ngrid, 1, 
		work, nfft, 1, 0, 
		(fftw_complex *)work, nfftc, 1, 0, 
		FFTW_MEASURE);
	ifft = fftw_plan_many_dft_c2r(3, ngrid, 1, 
		(fftw_complex *)work, nfftc, 1, 0, 
		work, nfft, 1, 0, 
		FFTW_MEASURE);

    /*	// The original interface, which only works if ngrid2 is tightly packed.
	fft = fftw_plan_dft_r2c_3d(ngrid[0], ngrid[1], ngrid[2],
			work, (fftw_complex *)work, FFTW_MEASURE);
	ifft = fftw_plan_dft_c2r_3d(ngrid[0], ngrid[1], ngrid[2], 
			(fftw_complex *)work, work, FFTW_MEASURE);
    */

    #else
	// If we wanted to split into 2D and 1D by hand (and therefore handle the OMP
	// aspects ourselves), then we need to have two plans each. 
	int nfft2[2], nfft2c[2];
	nfft2[0] = nfft2c[0] = ngrid[1];
	nfft2[1] = ngrid2;   // Since ngrid2 is always even, this will trick 
	nfft2c[1]= nfft2[1]/2;
	int ngridYZ[2]; 
	ngridYZ[0] = ngrid[1];
	ngridYZ[1] = ngrid[2];
	fftYZ = fftw_plan_many_dft_r2c(2, ngridYZ, 1, 
		work, nfft2, 1, 0, 
		(fftw_complex *)work, nfft2c, 1, 0, 
		FFTW_MEASURE);
	ifftYZ = fftw_plan_many_dft_c2r(2, ngridYZ, 1, 
		(fftw_complex *)work, nfft2c, 1, 0, 
		work, nfft2, 1, 0, 
		FFTW_MEASURE);

	// After we've done the 2D r2c FFT, we have to do the 1D c2c transform.
	// We'll plan to parallelize over Y, so that we're doing (ngrid[2]/2+1)
	// 1D FFTs at a time.
	// Elements in the X direction are separated by ngrid[1]*ngrid2/2 complex numbers.
	int ngridX = ngrid[0];
	fftX = fftw_plan_many_dft(1, &ngridX, (ngrid[2]/2+1), 
		(fftw_complex *)work, NULL, ngrid[1]*ngrid2/2, 1, 
		(fftw_complex *)work, NULL, ngrid[1]*ngrid2/2, 1, 
		-1, FFTW_MEASURE);
	ifftX = fftw_plan_many_dft(1, &ngridX, (ngrid[2]/2+1), 
		(fftw_complex *)work, NULL, ngrid[1]*ngrid2/2, 1, 
		(fftw_complex *)work, NULL, ngrid[1]*ngrid2/2, 1, 
		+1, FFTW_MEASURE);
    #endif

    fp = fopen(WISDOMFILE, "w");
    assert(fp!=NULL);
    fftw_export_wisdom_to_file(fp);
    fclose(fp);
    fprintf(stdout,"Done!\n"); fflush(NULL);
    FFTW.Stop();
    return;
}


void free_FFTW(fftw_plan &fft,  fftw_plan &fftYZ,  fftw_plan &fftX, 
               fftw_plan &ifft, fftw_plan &ifftYZ, fftw_plan &ifftX) {
    // Call all of the FFTW destroy routines
    #ifndef FFTSLAB
	fftw_destroy_plan(fft);
	fftw_destroy_plan(ifft);
    #else
	fftw_destroy_plan(fftYZ);
	fftw_destroy_plan(fftX);
	fftw_destroy_plan(ifftYZ);
	fftw_destroy_plan(ifftX);
    #endif
    #ifdef OPENMP
	#ifndef FFTSLAB
	    fftw_cleanup_threads();    
	#endif
    #endif
    return;
}

void FFT_Execute(fftw_plan fft, fftw_plan fftYZ, fftw_plan fftX, 
	    int ngrid[3], const int ngrid2, Float *work) {
    // Note that if FFTSLAB is not set, then the *work input is ignored!
    // Routine will use the array that was called for setup!
    // TODO: Might fix this behavior, but note alignment issues!
    FFTonly.Start();
    #ifndef FFTSLAB
	fftw_execute(fft);
    #else
	FFTyz.Start();
	// Then need to call this for every slab.  Can OMP these lines
	#pragma omp parallel for MY_SCHEDULE
	for (uint64 x=0; x<ngrid[0]; x++) 
	    fftw_execute_dft_r2c(fftYZ, work+x*ngrid[1]*ngrid2, 
	    	  (fftw_complex *)work+x*ngrid[1]*ngrid2/2);
	FFTyz.Stop();
	FFTx.Start();
	#pragma omp parallel for schedule(dynamic,1)
	for (uint64 y=0; y<ngrid[1]; y++) 
	    fftw_execute_dft(fftX, (fftw_complex *)work+y*ngrid2/2, 
	    	                   (fftw_complex *)work+y*ngrid2/2);
	FFTx.Stop();
    #endif
    FFTonly.Stop();
}

void IFFT_Execute(fftw_plan ifft, fftw_plan ifftYZ, fftw_plan ifftX,
	    int ngrid[3], const int ngrid2, Float *work) {
    // Note that if FFTSLAB is not set, then the *work input is ignored!
    // Routine will use the array that was called for setup!
    // TODO: Might fix this behavior, but note alignment issues!
    FFTonly.Start();
    #ifndef FFTSLAB
	fftw_execute(ifft);
    #else
	FFTx.Start();
	// Then need to call this for every slab.  Can OMP these lines
	#pragma omp parallel for schedule(dynamic,1)
	for (uint64 y=0; y<ngrid[1]; y++) 
	    fftw_execute_dft(ifftX, (fftw_complex *)work+y*ngrid2/2, 
	    	                   (fftw_complex *)work+y*ngrid2/2);
	FFTx.Stop();
	FFTyz.Start();
	#pragma omp parallel for MY_SCHEDULE
	for (uint64 x=0; x<ngrid[0]; x++) 
	    fftw_execute_dft_c2r(ifftYZ, 
	    	  (fftw_complex *)work+x*ngrid[1]*ngrid2/2,
				  work+x*ngrid[1]*ngrid2); 
	FFTyz.Stop();
    #endif
    FFTonly.Stop();
}

/* ======================================================================== */

// A very simple class to contain the input objects 
class Galaxy {
  public:
    Float x, y, z, w;
    uint64 index;
    Galaxy(Float a[4], uint64 i) { x = a[0]; y = a[1]; z = a[2]; w = a[3]; index = i; return; }
    ~Galaxy() { }
    // We'll want to be able to sort in 'x' order
    // bool operator < (const Galaxy& str) const { return (x < str.x); }
    // Sort in cell order
    bool operator < (const Galaxy& str) const { return (index < str.index); }
};

#include "merge_sort_omp.cpp"

/* ======================================================================== */

class Grid {
  public:
    // Inputs
    int ngrid[3];     // We might prefer a non-cubic box.  The cells are always cubic!
    Float max_sep;     // How much separation has already been built in.
    Float posmin[3];   // Including the border; we don't support periodic wrapping in CIC
    Float posmax[3];   // Including the border; we don't support periodic wrapping in CIC

    // Items to be computed
    Float posrange[3];    // The range of the padded box
    Float cell_size;      // The size of the cubic cells
    Float origin[3];      // The location of the origin in grid units.
    Float *xcell, *ycell, *zcell;   // The cell centers, relative to the origin

    // Storage for the r-space submatrices
    Float sep;		// The range of separations we'll be histogramming
    int csize[3];	// How many cells we must extract as a submatrix to do the histogramming.
    int csize3;		// The number of submatrix cells
    Float *cx_cell, *cy_cell, *cz_cell;   // The cell centers, relative to zero lag.
    Float *rnorm;	// The radius of each cell, in a flattened submatrix.

    // Storage for the k-space submatrices
    Float k_Nyq;        // The Nyquist frequency for our grid.
    Float kmax;		// The maximum wavenumber we'll use
    int ksize[3];	// How many cells we must extract as a submatrix to do the histogramming.
    int ksize3;		// The number of submatrix cells
    Float *kx_cell, *ky_cell, *kz_cell;   // The cell centers, relative to zero lag.
    Float *knorm;	// The wavenumber of each cell, in a flattened submatrix.
    Float *CICwindow;	// The inverse of the window function for the CIC cell assignment

    // The big grids
    int ngrid2;         // ngrid[2] padded out for the FFT work
    uint64 ngrid3;	// The total number of FFT grid cells
    Float *dens; 	// The density field, in a flattened grid
    Float *densFFT; 	// The FFT of the density field, in a flattened grid.
    Float *work; 	// Work space for each (ell,m), in a flattened grid.

    int cnt;		// The number of galaxies read in.
    Float Pshot;	// The sum of squares of the weights, which is the shot noise for P_0.

    // Positions need to arrive in a coordinate system that has the observer at the origin

    ~Grid() {
        if (dens!=NULL) free(dens);
        if (densFFT!=NULL) free(densFFT);
        if (work!=NULL) free(work);
        free(zcell);
        free(ycell);
        free(xcell);
        free(rnorm);
        free(cx_cell);
        free(cy_cell);
        free(cz_cell);
        free(knorm);
        free(kx_cell);
        free(ky_cell);
        free(kz_cell);
	free(CICwindow);
	// *densFFT and *work are freed in the correlate() routine.
    }

    Grid(const char filename[], int _ngrid[3], Float _cell, Float _sep, int qperiodic) {
	// This constructor is rather elaborate, but we're going to do most of the setup.
	// filename and filename2 are the input particles.
	// filename2==NULL will skip that one.
	// _sep is used here simply to adjust the box size if needed.
	// qperiodic flag will configure for periodic BC

	// Have to set these to null so that the initialization will work.
	dens = densFFT = work = NULL;
	rnorm = knorm = CICwindow = NULL;

        // Open a binary input file
	Setup.Start();
	FILE *fp = fopen(filename, "rb");
	assert(fp!=NULL);

	for (int j=0; j<3; j++) ngrid[j] = _ngrid[j];
	assert(ngrid[0]>0&&ngrid[0]<1e4);
	assert(ngrid[1]>0&&ngrid[1]<1e4);
	assert(ngrid[2]>0&&ngrid[2]<1e4);

	Float TOOBIG = 1e10;
	// This header is 64 bytes long.
	// Read posmin[3], posmax[3], max_sep, blank8;
	double tmp[4]; 
	int nread;
	nread=fread(tmp, sizeof(double), 3, fp); assert(nread==3);
	for (int j=0; j<3; j++) { posmin[j]=tmp[j]; assert(fabs(posmin[0])<TOOBIG); }
	nread=fread(tmp, sizeof(double), 3, fp); assert(nread==3);
	for (int j=0; j<3; j++) { posmax[j]=tmp[j]; assert(fabs(posmax[0])<TOOBIG); }
	nread=fread(tmp, sizeof(double), 1, fp); assert(nread==1);
	max_sep = tmp[0]; assert(max_sep>=0&&max_sep<TOOBIG);
	nread=fread(tmp, sizeof(double), 1, fp); assert(nread==1); // Not used, just for alignment
	fclose(fp);

	// If we're going to permute the axes, change here and in add_particles_to_grid().
	// The answers should be unchanged under permutation
	// std::swap(posmin[0], posmin[1]); std::swap(posmax[0], posmax[1]);
	// std::swap(posmin[2], posmin[1]); std::swap(posmax[2], posmax[1]);

        // If the user wants periodic BC, then we can ignore separation issues.
	if (qperiodic) max_sep = (posmax[0]-posmin[0])*100.0;

	// If the user asked for a larger separation than what was planned in the 
	// input positions, then we can accomodate.  Add the extra padding to posrange;
	// don't change posmin, since that changes grid registration.
	Float extra_pad = 0.0;
	if (_sep>max_sep) {
	    extra_pad = _sep-max_sep;
	    max_sep = _sep;
	}
	sep = -1;   // Just as a test that setup() got run

	// Compute the box size required in each direction
	for (int j=0; j<3; j++) { 
	    posmax[j] += extra_pad;
	    posrange[j]=posmax[j]-posmin[j];    
	    assert(posrange[j]>0.0); 
	}

	if (qperiodic||_cell<=0) {
	    // We need to compute the cell size
	    // We've been given 3 ngrid and we have the bounding box.
	    // Need to pick the most conservative choice
	    // This is always required in the periodic case
	    cell_size = std::max(posrange[0]/ngrid[0], 
	    	        std::max(posrange[1]/ngrid[1], posrange[2]/ngrid[2]));
	} else {
	    // We've been given a cell size and a grid.  Need to assure it is ok.
	    cell_size = _cell;
	    assert(cell_size*ngrid[0]>posrange[0]);
	    assert(cell_size*ngrid[1]>posrange[1]);
	    assert(cell_size*ngrid[2]>posrange[2]);
	}

	fprintf(stdout, "# Reading file %s.  max_sep=%f\n", filename, max_sep);
	fprintf(stdout, "# Adopting cell_size=%f for ngrid=%d, %d, %d\n", 
		cell_size, ngrid[0], ngrid[1], ngrid[2]);
	fprintf(stdout, "# Adopted boxsize: %6.1f %6.1f %6.1f\n", 
		cell_size*ngrid[0], cell_size*ngrid[1], cell_size*ngrid[2]);
	fprintf(stdout, "# Input pos range: %6.1f %6.1f %6.1f\n", 
		posrange[0], posrange[1], posrange[2]);
	fprintf(stdout, "# Minimum ngrid=%d, %d, %d\n", int(ceil(posrange[0]/cell_size)),
		int(ceil(posrange[1]/cell_size)), int(ceil(posrange[2]/cell_size)));
		
	// ngrid2 pads out the array for the in-place FFT.
	// The default 3d FFTW format must have the following:
        ngrid2 = (ngrid[2]/2+1)*2;  // For the in-place FFT
	#ifdef FFTSLAB
	    // That said, the rest of the code should work even extra space is used.
	    // Some operations will blindly apply to the pad cells, but that's ok.
	    // In particular, we might consider having ngrid2 be evenly divisible by
	    // the critical alignment stride (32 bytes for AVX, but might be more for cache lines) 
	    // or even by a full PAGE for NUMA memory.  Doing this *will* force a more 
	    // complicated FFT, but at least for the NUMA case this is desired: we want 
	    // to force the 2D FFT to run on its socket, and only have the last 1D FFT
	    // crossing sockets.  Re-using FFTW plans requires the consistent memory alignment.
	    #define FFT_ALIGN 16		
	    // This is in units of Floats.  16 doubles is 1024 bits.
	    ngrid2 = FFT_ALIGN*(ngrid2/FFT_ALIGN+1);
	#endif
	assert(ngrid2%2==0);
	fprintf(stdout, "# Using ngrid2=%d for FFT r2c padding\n", ngrid2);
	ngrid3 = (uint64)ngrid[0]*ngrid[1]*ngrid2;

	// Convert origin to grid units
	if (qperiodic) {
	    // In this case, we'll place the observer centered in the grid, but 
	    // then displaced far away in the -x direction
	    for (int j=0;j<3;j++) origin[j] = ngrid[j]/2.0;
	    origin[0] -= ngrid[0]*1e6;	// Observer far away!
	} else {
	    for (int j=0;j<3;j++) origin[j] = (0.0-posmin[j])/cell_size;
	}


	// Allocate xcell, ycell, zcell to [ngrid]
	int err;
	err=posix_memalign((void **) &xcell, PAGE, sizeof(Float)*ngrid[0]+PAGE); assert(err==0);
	err=posix_memalign((void **) &ycell, PAGE, sizeof(Float)*ngrid[1]+PAGE); assert(err==0);
	err=posix_memalign((void **) &zcell, PAGE, sizeof(Float)*ngrid[2]+PAGE); assert(err==0);
	assert(xcell!=NULL); assert(ycell!=NULL); assert(zcell!=NULL);
	// Now set up the cell centers relative to the origin, in grid units
	for (int j=0; j<ngrid[0]; j++) xcell[j] = 0.5+j-origin[0];
	for (int j=0; j<ngrid[1]; j++) ycell[j] = 0.5+j-origin[1];
	for (int j=0; j<ngrid[2]; j++) zcell[j] = 0.5+j-origin[2];
	Setup.Stop(); 

	// Allocate dens to [ngrid**2*ngrid2] and set it to zero
	initialize_matrix(dens, ngrid3, ngrid[0]);
	return;
    }

/* ------------------------------------------------------------------- */

    void read_galaxies(const char filename[], const char filename2[], int qperiodic) {
	// Read to the end of the file, bringing in x,y,z,w points.
	// Bin them onto the grid.
	// We're setting up a large buffer to read in the galaxies.
	// Will reset the buffer periodically, just to limit the size.
	double tmp[8];
	cnt = 0;
	uint64 index;
	Float totw = 0.0, totwsq = 0.0;
	// Set up a small buffer, just to reduce the calls to fread, which seem to be slow
	// on some machines.
	#define BUFFERSIZE 512
	double buffer[BUFFERSIZE], *b;
	#define MAXGAL 1000000
	std::vector<Galaxy> gal;
	gal.reserve(MAXGAL);    // Just to cut down on thrashing; it will expand as needed

	IO.Start();
	for (int file=0; file<2; file++) {
	    char *fn;
	    int thiscnt=0;
	    if (file==0) fn = (char *)filename; else fn = (char *)filename2;
	    if (fn==NULL) continue;   // No file!
	    fprintf(stdout, "# Reading from file %d named %s\n", file, fn);
	    FILE *fp = fopen(fn,  "rb");
	    assert(fp!=NULL);
	    int nread=fread(tmp, sizeof(double), 8, fp); assert(nread==8); // Skip the header
	    while ((nread=fread(&buffer, sizeof(double), BUFFERSIZE, fp))>0) {
		b=buffer;
		for (int j=0; j<nread; j+=4,b+=4) {
		    index=change_to_grid_coords(b);
		    gal.push_back(Galaxy(b,index));
		    thiscnt++; totw += b[3]; totwsq += b[3]*b[3];
		    if (gal.size()>=MAXGAL) {
			IO.Stop(); add_to_grid(gal); IO.Start();
		    }
		}
		if (nread!=BUFFERSIZE) break;
	    }
	    cnt += thiscnt;
	    fprintf(stdout, "# Found %d galaxies in this file\n", thiscnt);
	    fclose(fp);
	}
	IO.Stop();
	// Add the remaining galaxies to the grid
	add_to_grid(gal);

	fprintf(stdout, "# Found %d particles. Total weight %10.4e.\n", cnt, totw);
	Float totw2 = sum_matrix(dens, ngrid3, ngrid[0]);
	fprintf(stdout, "# Sum of grid is %10.4e (delta = %10.4e)\n", totw2, totw2-totw);
	if (qperiodic==2) {
	     // We're asked to set the mean to zero
	     Float mean = totw/ngrid[0]/ngrid[1]/ngrid[2];
	     addscalarto_matrix(dens, -mean, ngrid3, ngrid[0]);
 	     fprintf(stdout, "# Subtracting mean cell density %10.4e\n", mean);
	}

	Float sumsq_dens = sumsq_matrix(dens, ngrid3, ngrid[0]);
	fprintf(stdout, "# Sum of squares of density = %14.7e\n", sumsq_dens);
	Pshot = totwsq;
	fprintf(stdout, "# Sum of squares of weights (divide by I for Pshot) = %14.7e\n", Pshot);
	// When run with N=D-R, this divided by I would be the shot noise.

	// Meanwhile, an estimate of I when running with only R is 
	// (sum of R^2)/Vcell - (11/20)**3*(sum_R w^2)/Vcell
	// The latter is correcting the estimate for shot noise
	// The 11/20 factor is for triangular cloud in cell.
	#ifndef NEAREST_CELL
	#ifdef WAVELET
	    fprintf(stdout, "# Using D12 wavelet\n");
	#else
	    totwsq *= 0.55*0.55*0.55;
	    fprintf(stdout, "# Using triangular cloud-in-cell\n");
	#endif
	#else
	    fprintf(stdout, "# Using nearest cell method\n");
	#endif
	Float Vcell = cell_size*cell_size*cell_size;
	fprintf(stdout, "# Estimate of I (denominator) = %14.7e - %14.7e = %14.7e\n", 
		sumsq_dens/Vcell, totwsq/Vcell, (sumsq_dens-totwsq)/Vcell);

	// In the limit of infinite homogeneous particles in a periodic box:
	// If W=sum(w), then each particle has w = W/N.  totwsq = N*(W/N)^2 = W^2/N.
	// Meanwhile, each cell has density (W/N)*(N/Ncell) = W/Ncell. 
	// sumsq_dens/Vcell = W^2/(Ncell*Vcell) = W^2/V.
	// Hence the real shot noise is V/N = 1/n.
	return;
    }

/* ------------------------------------------------------------------- */

    void add_to_grid(std::vector<Galaxy> &gal) {
	// Given a set of Galaxies, add them to the grid and then reset the list
	CIC.Start();
	const int galsize = gal.size();

    #ifdef DEPRICATED
	// This works, but appears to be slower
	for (int j=0; j<galsize; j++) add_particle_to_grid(gal[j]);
    #else
	// If we're parallelizing this, then we need to keep the threads from stepping
	// on each other.  Do this in slabs, but with only every third slab active at 
	// any time.

	// Let's sort the particles by x.
	// Need to supply an equal amount of temporary space to merge sort.
	// Do this by another vector.
	std::vector<Galaxy> tmp;
	tmp.reserve(galsize);    
	mergesort_parallel_omp(gal.data(), galsize, tmp.data(), omp_get_max_threads());
	// This just falls back to std::sort if omp_get_max_threads==1

	// Now we need to find the starting point of each slab
	// Galaxies between N and N+1 should be in indices [first[N], first[N+1]).
	// That means that first[N] should be the index of the first galaxy to exceed N. 
	int first[ngrid[0]+1], ptr=0;
	for (int j=0; j<galsize; j++) 
	    while (gal[j].x>ptr) 
		first[ptr++] = j;
	for (;ptr<=ngrid[0];ptr++) first[ptr]=galsize;

	// Now, we'll loop, with each thread in charge of slab x.
	// Not bothering with NUMA issues.  a) Most of the time is spent waiting for 
	// memory to respond, not actually piping between processors.  b) Adjacent 
	// slabs may not be on the same memory bank anyways.  Keep it simple.
	int slabset = 3;
	#ifdef WAVELET
	    slabset = WCELLS;
	#endif
	for (int mod=0; mod<slabset; mod++) {
	    #pragma omp parallel for schedule(dynamic,1)
	    for (int x=mod; x<ngrid[0]; x+=slabset) {
		// For each slab, insert these particles
		for (int j=first[x]; j<first[x+1]; j++) add_particle_to_grid(gal[j]);
	    }
	}
    #endif
	gal.clear();
	gal.reserve(MAXGAL);    // Just to check!
	CIC.Stop();
	return;
    }

/* ------------------------------------------------------------------- */

    inline uint64 change_to_grid_coords(Float tmp[4]) {
	// Given tmp[4] = x,y,z,w,
	// Modify to put them in box coordinates.
	// We'll have no use for the original coordinates!
	// tmp[3] (w) is unchanged
	tmp[0] = (tmp[0]-posmin[0])/cell_size;
	tmp[1] = (tmp[1]-posmin[1])/cell_size;
	tmp[2] = (tmp[2]-posmin[2])/cell_size;
	uint64 ix = floor(tmp[0]);
	uint64 iy = floor(tmp[1]);
	uint64 iz = floor(tmp[2]);
	return (iz)+ngrid2*((iy)+(ix)*ngrid[1]);
    }

    void add_particle_to_grid(Galaxy g) {
        // Add one particle to the density grid. 
	// This does a 27-point triangular cloud-in-cell, unless one invokes NEAREST_CELL.
	uint64 index;   // Trying not to assume that ngrid**3 won't spill 32-bits.
	uint64 ix = floor(g.x);
	uint64 iy = floor(g.y);
	uint64 iz = floor(g.z);

	// If we're just doing nearest cell.
	#ifdef NEAREST_CELL
	    index = (iz)+ngrid2*((iy)+(ix)*ngrid[1]);
	    dens[index] += g.w;
	    return;
	#endif

	#ifdef WAVELET
	    // In the wavelet version, we truncate to 1/WAVESAMPLE resolution in each
	    // cell and use a lookup table.  Table is set up so that each sub-cell 
	    // resolution has the values for the various integral cell offsets contiguous
	    // in memory.
	    uint64 sx = floor((g.x-ix)*WAVESAMPLE);
	    uint64 sy = floor((g.y-iy)*WAVESAMPLE);
	    uint64 sz = floor((g.z-iz)*WAVESAMPLE);
	    const Float *xwave = wave+sx*WCELLS;
	    const Float *ywave = wave+sy*WCELLS;
	    const Float *zwave = wave+sz*WCELLS;
	    // This code does periodic wrapping
	    const uint64 ng0 = ngrid[0];
	    const uint64 ng1 = ngrid[1];
	    const uint64 ng2 = ngrid[2];
	    // Offset to the lower-most cell, taking care to handle unsigned int
	    ix = (ix+ng0+WMIN)%ng0;
	    iy = (iy+ng1+WMIN)%ng1;
	    iz = (iz+ng2+WMIN)%ng2;
	    Float *px = dens+ngrid2*ng1*ix;
	    for (int ox=0; ox<WCELLS; ox++, px+=ngrid2*ng1) {
		if (ix+ox==ng0) px -= ng0*ng1*ngrid2;  // Periodic wrap in X
		Float Dx = xwave[ox]*g.w;
		Float *py = px + iy*ngrid2;
		for (int oy=0; oy<WCELLS; oy++, py+=ngrid2) {
		    if (iy+oy==ng1) py -= ng1*ngrid2;  // Periodic wrap in Y
		    Float *pz = py+iz;
		    Float Dxy = Dx*ywave[oy];
		    if (iz+WCELLS>ng2) {     // Z Wrap is needed
			for (int oz=0; oz<WCELLS; oz++) {
			    if (iz+oz==ng2) pz -= ng2;  // Periodic wrap in Z
			    pz[oz] += zwave[oz]*Dxy;
			}
		    } else {
			for (int oz=0; oz<WCELLS; oz++) pz[oz] += zwave[oz]*Dxy;
		    }
		}
	    }
	    return;
	#endif

	// Now to Cloud-in-Cell
	Float rx = g.x-ix;
	Float ry = g.y-iy;
	Float rz = g.z-iz;
	//
	Float xm = 0.5*(1-rx)*(1-rx)*g.w;
	Float xp = 0.5*rx*rx*g.w;
	Float x0 = (0.5+rx-rx*rx)*g.w;
	Float ym = 0.5*(1-ry)*(1-ry);
	Float yp = 0.5*ry*ry;
	Float y0 = 0.5+ry-ry*ry;
	Float zm = 0.5*(1-rz)*(1-rz);
	Float zp = 0.5*rz*rz;
	Float z0 = 0.5+rz-rz*rz;
	//
	if (ix==0||ix==ngrid[0]-1 || iy==0||iy==ngrid[1]-1 || iz==0||iz==ngrid[2]-1) {
	    // This code does periodic wrapping
	    const uint64 ng0 = ngrid[0];
	    const uint64 ng1 = ngrid[1];
	    const uint64 ng2 = ngrid[2];
	    ix += ngrid[0];   // Just to put away any fears of negative mods
	    iy += ngrid[1];
	    iz += ngrid[2];
	    const uint64 izm = (iz-1)%ng2;
	    const uint64 iz0 = (iz  )%ng2;
	    const uint64 izp = (iz+1)%ng2;
	    //
	    index = ngrid2*(((iy-1)%ng1)+((ix-1)%ng0)*ng1);
	    dens[index+izm] += xm*ym*zm;
	    dens[index+iz0] += xm*ym*z0;
	    dens[index+izp] += xm*ym*zp;
	    index = ngrid2*(((iy  )%ng1)+((ix-1)%ng0)*ng1);
	    dens[index+izm] += xm*y0*zm;
	    dens[index+iz0] += xm*y0*z0;
	    dens[index+izp] += xm*y0*zp;
	    index = ngrid2*(((iy+1)%ng1)+((ix-1)%ng0)*ng1);
	    dens[index+izm] += xm*yp*zm;
	    dens[index+iz0] += xm*yp*z0;
	    dens[index+izp] += xm*yp*zp;
	    //
	    index = ngrid2*(((iy-1)%ng1)+((ix  )%ng0)*ng1);
	    dens[index+izm] += x0*ym*zm;
	    dens[index+iz0] += x0*ym*z0;
	    dens[index+izp] += x0*ym*zp;
	    index = ngrid2*(((iy  )%ng1)+((ix  )%ng0)*ng1);
	    dens[index+izm] += x0*y0*zm;
	    dens[index+iz0] += x0*y0*z0;
	    dens[index+izp] += x0*y0*zp;
	    index = ngrid2*(((iy+1)%ng1)+((ix  )%ng0)*ng1);
	    dens[index+izm] += x0*yp*zm;
	    dens[index+iz0] += x0*yp*z0;
	    dens[index+izp] += x0*yp*zp;
	    //
	    index = ngrid2*(((iy-1)%ng1)+((ix+1)%ng0)*ng1);
	    dens[index+izm] += xp*ym*zm;
	    dens[index+iz0] += xp*ym*z0;
	    dens[index+izp] += xp*ym*zp;
	    index = ngrid2*(((iy  )%ng1)+((ix+1)%ng0)*ng1);
	    dens[index+izm] += xp*y0*zm;
	    dens[index+iz0] += xp*y0*z0;
	    dens[index+izp] += xp*y0*zp;
	    index = ngrid2*(((iy+1)%ng1)+((ix+1)%ng0)*ng1);
	    dens[index+izm] += xp*yp*zm;
	    dens[index+iz0] += xp*yp*z0;
	    dens[index+izp] += xp*yp*zp;
	} else {
	    // This code is faster, but doesn't do periodic wrapping
	    index = (iz-1)+ngrid2*((iy-1)+(ix-1)*ngrid[1]);
	    dens[index++] += xm*ym*zm;
	    dens[index++] += xm*ym*z0;
	    dens[index]   += xm*ym*zp;
	    index += ngrid2-2;   // Step to the next row in y
	    dens[index++] += xm*y0*zm;
	    dens[index++] += xm*y0*z0;
	    dens[index]   += xm*y0*zp;
	    index += ngrid2-2;   // Step to the next row in y
	    dens[index++] += xm*yp*zm;
	    dens[index++] += xm*yp*z0;
	    dens[index]   += xm*yp*zp;
	    index = (iz-1)+ngrid2*((iy-1)+ix*ngrid[1]);
	    dens[index++] += x0*ym*zm;
	    dens[index++] += x0*ym*z0;
	    dens[index]   += x0*ym*zp;
	    index += ngrid2-2;   // Step to the next row in y
	    dens[index++] += x0*y0*zm;
	    dens[index++] += x0*y0*z0;
	    dens[index]   += x0*y0*zp;
	    index += ngrid2-2;   // Step to the next row in y
	    dens[index++] += x0*yp*zm;
	    dens[index++] += x0*yp*z0;
	    dens[index]   += x0*yp*zp;
	    index = (iz-1)+ngrid2*((iy-1)+(ix+1)*ngrid[1]);
	    dens[index++] += xp*ym*zm;
	    dens[index++] += xp*ym*z0;
	    dens[index]   += xp*ym*zp;
	    index += ngrid2-2;   // Step to the next row in y
	    dens[index++] += xp*y0*zm;
	    dens[index++] += xp*y0*z0;
	    dens[index]   += xp*y0*zp;
	    index += ngrid2-2;   // Step to the next row in y
	    dens[index++] += xp*yp*zm;
	    dens[index++] += xp*yp*z0;
	    dens[index]   += xp*yp*zp;
	}
    }

/* ------------------------------------------------------------------- */

    Float setup_corr(Float _sep, Float _kmax) {
	// Set up the sub-matrix information, assuming that we'll extract 
	// -sep..+sep cells around zero-lag.
	// _sep<0 causes a default to the value in the file.
	Setup.Start();
	if (_sep<0) sep = max_sep;
	    else sep = _sep;
	fprintf(stdout,"# Chosen separation %f vs max %f\n",sep, max_sep);
	assert(sep<=max_sep);

	int sep_cell = ceil(sep/cell_size);
	csize[0] = 2*sep_cell+1;
	csize[1] = csize[2] = csize[0];
	assert(csize[0]%2==1); assert(csize[1]%2==1); assert(csize[2]%2==1);
	csize3 = csize[0]*csize[1]*csize[2];
	// Allocate corr_cell to [csize] and rnorm to [csize**3]
	int err;
	err=posix_memalign((void **) &cx_cell, PAGE, sizeof(Float)*csize[0]+PAGE); assert(err==0);
	err=posix_memalign((void **) &cy_cell, PAGE, sizeof(Float)*csize[1]+PAGE); assert(err==0);
	err=posix_memalign((void **) &cz_cell, PAGE, sizeof(Float)*csize[2]+PAGE); assert(err==0);
	initialize_matrix(rnorm, csize3, csize[0]);

	// Normalizing by cell_size just so that the Ylm code can do the wide-angle
	// corrections in the same units.
	for (int i=0; i<csize[0]; i++) cx_cell[i] = cell_size*(i-sep_cell);
	for (int i=0; i<csize[1]; i++) cy_cell[i] = cell_size*(i-sep_cell);
	for (int i=0; i<csize[2]; i++) cz_cell[i] = cell_size*(i-sep_cell);

	for (uint64 i=0; i<csize[0]; i++)
	    for (int j=0; j<csize[1]; j++)
		for (int k=0; k<csize[2]; k++)
		    rnorm[k+csize[2]*(j+i*csize[1])] = cell_size*sqrt( 
		    	 (i-sep_cell)*(i-sep_cell)
		    	+(j-sep_cell)*(j-sep_cell)
		    	+(k-sep_cell)*(k-sep_cell));
	fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n", sep_cell);

	// Our box has cubic-sized cells, so k_Nyquist is the same in all directions
	// The spacing of modes is therefore 2*k_Nyq/ngrid
	k_Nyq = M_PI/cell_size;
	kmax = _kmax;
	fprintf(stdout, "# Storing wavenumbers up to %6.4f, with k_Nyq = %6.4f\n", kmax, k_Nyq);
	for (int i=0; i<3; i++) ksize[i] = 2*ceil(kmax/(2.0*k_Nyq/ngrid[i]))+1;
	assert(ksize[0]%2==1); assert(ksize[1]%2==1); assert(ksize[2]%2==1);
	for (int i=0; i<3; i++) if (ksize[i]>ngrid[i]) {
	    ksize[i] = 2*floor(ngrid[i]/2)+1;
	    fprintf(stdout, "# WARNING: Requested wavenumber is too big.  Truncating ksize[%d] to %d\n", i, ksize[i]);
	}

	ksize3 = ksize[0]*ksize[1]*ksize[2];
	// Allocate kX_cell to [ksize] and knorm to [ksize**3]
	err=posix_memalign((void **) &kx_cell, PAGE, sizeof(Float)*ksize[0]+PAGE); assert(err==0);
	err=posix_memalign((void **) &ky_cell, PAGE, sizeof(Float)*ksize[1]+PAGE); assert(err==0);
	err=posix_memalign((void **) &kz_cell, PAGE, sizeof(Float)*ksize[2]+PAGE); assert(err==0);
	initialize_matrix(knorm, ksize3, ksize[0]);
	initialize_matrix(CICwindow, ksize3, ksize[0]);

	for (int i=0; i<ksize[0]; i++) kx_cell[i] = (i-ksize[0]/2)*2.0*k_Nyq/ngrid[0];
	for (int i=0; i<ksize[1]; i++) ky_cell[i] = (i-ksize[1]/2)*2.0*k_Nyq/ngrid[1];
	for (int i=0; i<ksize[2]; i++) kz_cell[i] = (i-ksize[2]/2)*2.0*k_Nyq/ngrid[2];

	for (uint64 i=0; i<ksize[0]; i++)
	    for (int j=0; j<ksize[1]; j++)
		for (int k=0; k<ksize[2]; k++) {
		    knorm[k+ksize[2]*(j+i*ksize[1])] = sqrt( kx_cell[i]*kx_cell[i]
		    	+ky_cell[j]*ky_cell[j] +kz_cell[k]*kz_cell[k]);
		    // For TSC, the square window is 1-sin^2(kL/2)+2/15*sin^4(kL/2)
		    Float sinkxL = sin(kx_cell[i]*cell_size/2.0);
		    Float sinkyL = sin(ky_cell[j]*cell_size/2.0);
		    Float sinkzL = sin(kz_cell[k]*cell_size/2.0);
		    sinkxL *= sinkxL;
		    sinkyL *= sinkyL;
		    sinkzL *= sinkzL;
		    Float Wx, Wy, Wz;
		    Wx = 1-sinkxL+2.0/15.0*sinkxL*sinkxL;
		    Wy = 1-sinkyL+2.0/15.0*sinkyL*sinkyL;
		    Wz = 1-sinkzL+2.0/15.0*sinkzL*sinkzL;
		    Float window = Wx*Wy*Wz;   // This is the square of the window
		    #ifdef NEAREST_CELL
			// For this case, the window is unity
			window = 1.0;
		    #endif
		    #ifdef WAVELET
			// For this case, the window is unity
			window = 1.0;
		    #endif
		    CICwindow[k+ksize[2]*(j+i*ksize[1])] = 1.0/window;
		    // We will divide the power spectrum by the square of the window
		}


	fprintf(stdout, "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n", 
		ksize[0]/2, ksize[1]/2, ksize[2]/2);

	Setup.Stop();
	return sep;
    }

    void print_submatrix(Float *m, int n, int p, FILE *fp, Float norm) {
        // Print the inner part of a matrix(n,n,n) for debugging
	int mid = n/2;
	assert(p<=mid);
	for (int i=-p; i<=p; i++)
	    for (int j=-p; j<=p; j++) {
		fprintf(fp, "%2d %2d", i, j);
		for (int k=-p; k<=p; k++) {
		    // We want to print mid+i, mid+j, mid+k
		    fprintf(fp, " %12.8g", m[((mid+i)*n+(mid+j))*n+mid+k]*norm);
		}
		fprintf(fp, "\n");
	    }
	return;
    }

/* ------------------------------------------------------------------- */

    void correlate(int maxell, Histogram &h, Histogram &kh, int wide_angle_exponent) {
	// Here's where most of the work occurs.
	// This computes the correlations for each ell, summing over m,
	// and then histograms the result.
	void makeYlm(Float *work, int ell, int m, int n[3], int n1, 
			Float *xcell, Float *ycell, Float *zcell, Float *dens, int exponent);

	// Multiply total by 4*pi, to match SE15 normalization
	// Include the FFTW normalization
	Float norm = 4.0*M_PI/ngrid[0]/ngrid[1]/ngrid[2];
	Float Pnorm = 4.0*M_PI;
	assert(sep>0);    // This is a check that the submatrix got set up.

	// Allocate the work matrix and load it with the density
	// We do this here so that the array is touched before FFT planning
	initialize_matrix_by_copy(work, ngrid3, ngrid[0], dens);

	// Allocate total[csize**3] and corr[csize**3]
	Float *total=NULL;  initialize_matrix(total,  csize3, csize[0]);
	Float *corr=NULL;   initialize_matrix(corr,   csize3, csize[0]);
	Float *ktotal=NULL; initialize_matrix(ktotal, ksize3, ksize[0]);
	Float *kcorr=NULL;  initialize_matrix(kcorr,  ksize3, ksize[0]);

	/* Setup FFTW */
	fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
	setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, ngrid, ngrid2, work);

	// FFTW might have destroyed the contents of work; need to restore work[]==dens[]
	// So far, I haven't seen this happen.
	if (dens[1]!=work[1] || dens[1+ngrid[2]]!=work[1+ngrid[2]] 
			     || dens[ngrid3-1]!=work[ngrid3-1]) {
	    fprintf(stdout, "Restoring work matrix\n");
	    Init.Start();
	    copy_matrix(work, dens, ngrid3, ngrid[0]);
	    Init.Stop();
	}

	Correlate.Start();    // Starting the main work
        // Now compute the FFT of the density field and conjugate it
	// FFT(work) in place and conjugate it, storing in densFFT
	fprintf(stdout,"# Computing the density FFT..."); fflush(NULL);
	FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

	Correlate.Stop();    // We're tracking initialization separately
	initialize_matrix_by_copy(densFFT, ngrid3, ngrid[0], work);
	fprintf(stdout,"Done!\n"); fflush(NULL);
	Correlate.Start();

	// Let's try a check as well -- convert with the 3D code and compare
	/* copy_matrix(work, dens, ngrid3, ngrid[0]);
	fftw_execute(fft);
	for (uint64 j=0; j<ngrid3; j++) 
	    if (densFFT[j]!=work[j]) {
		int z = j%ngrid2;
		int y = j/ngrid2; y=y%ngrid2;
		int x = j/ngrid[1]/ngrid2;
		printf("%d %d %d  %f  %f\n", x, y, z, densFFT[j], work[j]);
	    }
	*/

	/* ------------ Loop over ell & m --------------- */
	// Loop over each ell to compute the anisotropic correlations
	for (int ell=0; ell<=maxell; ell+=2) {
	    // Initialize the submatrix
	    set_matrix(total,0.0, csize3, csize[0]);
	    set_matrix(ktotal,0.0, ksize3, ksize[0]);
	    // Loop over m
	    for (int m=-ell; m<=ell; m++) {
		fprintf(stdout,"# Computing %d %2d...", ell, m);
		// Create the Ylm matrix times dens
	        makeYlm(work, ell, m, ngrid, ngrid2, xcell, ycell, zcell, dens, -wide_angle_exponent);
		fprintf(stdout,"Ylm...");

		// FFT in place
		FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

		// Multiply by conj(densFFT), as complex numbers
		AtimesB.Start();
		multiply_matrix_with_conjugation((Complex *)work, 
				(Complex *)densFFT, ngrid3/2, ngrid[0]);
		AtimesB.Stop();

		// Extract the anisotropic power spectrum
		// Load the Ylm's and include the CICwindow correction
		makeYlm(kcorr, ell, m, ksize, ksize[2], kx_cell, ky_cell, kz_cell, CICwindow, wide_angle_exponent);
		// Multiply these Ylm by the power result, and then add to total.
		extract_submatrix_C2R(ktotal, kcorr, ksize, (Complex *)work, ngrid, ngrid2);

		// iFFT the result, in place
		IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
		fprintf(stdout,"FFT...");

		// Create Ylm for the submatrix that we'll extract for histogramming
		// The extra multiplication by one here is of negligible cost, since
		// this array is so much smaller than the FFT grid.
		makeYlm(corr, ell, m, csize, csize[2], cx_cell, cy_cell, cz_cell, NULL, wide_angle_exponent);

		// Multiply these Ylm by the correlation result, and then add to total.
		extract_submatrix(total, corr, csize, work, ngrid, ngrid2);

		fprintf(stdout,"Done!\n");
	    }

	    Extract.Start();
	    scale_matrix(total, norm, csize3, csize[0]);
	    scale_matrix(ktotal, Pnorm, ksize3, ksize[0]);
	    Extract.Stop();
	    // Histogram total by rnorm
	    Hist.Start();
	    h.histcorr(ell, csize3, rnorm, total);
	    kh.histcorr(ell, ksize3, knorm, ktotal);
	    Hist.Stop();

	}

	/* ------------------- Clean up -------------------*/
	// Free densFFT and Ylm
	free(corr);
	free(total);
	free(kcorr);
	free(ktotal);
	free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

	Correlate.Stop();
    }
};    // end Grid

/* =========================================================================== */

void ReportTimes(FILE *fp, uint64 nfft, uint64 ngrid3, int cnt) {
    fflush(NULL);
    fprintf(fp, "#\n# Timing Report: \n");
    fprintf(fp, "# Setup time:       %8.4f s\n", Setup.Elapsed());
    fprintf(fp, "# I/O time:         %8.4f s, %6.3f Mparticles/sec, %6.2f MB/sec Read\n", 
    	IO.Elapsed(), cnt/IO.Elapsed()/1e6, cnt/IO.Elapsed()*32.0/1e6);
    fprintf(fp, "# CIC Grid time:    %8.4f s, %6.3f Mparticles/sec, %6.2f GB/sec\n", CIC.Elapsed(),
    	cnt/CIC.Elapsed()/1e6, 1e-9*cnt/CIC.Elapsed()*27.0*2.0*sizeof(Float));
    fprintf(fp, "#        Sorting:   %8.4f s\n", Sorting.Elapsed());
    fprintf(fp, "#        Merging:   %8.4f s\n", Merging.Elapsed());
    fprintf(fp, "#            CIC:   %8.4f s\n", CIC.Elapsed()-Merging.Elapsed()-Sorting.Elapsed());
    fprintf(fp, "# FFTW Prep time:   %8.4f s\n", FFTW.Elapsed());
    fprintf(fp, "# Array Init time:  %8.4f s, %6.3f GB/s\n", Init.Elapsed(),
    		1e-9*ngrid3*sizeof(Float)*5/Init.Elapsed());
    fprintf(fp, "# Correlate time:   %8.4f s\n", Correlate.Elapsed());
    // Expecting 6 Floats of load/store
    fprintf(fp, "#       FFT time:   %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n", 
        FFTonly.Elapsed(), 
    	nfft/1e6/FFTonly.Elapsed(), 
    	nfft/1e9/FFTonly.Elapsed()*6.0*sizeof(Float),
    	nfft/1e6/FFTonly.Elapsed()*2.5*log(ngrid3)/log(2)/1e3); 
#ifdef FFTSLAB
    fprintf(fp, "#     FFTyz time:   %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n", 
        FFTyz.Elapsed(), 
    	nfft/1e6/FFTyz.Elapsed()*2.0/3.0, 
    	nfft/1e9/FFTyz.Elapsed()*6.0*sizeof(Float)*2.0/3.0,
    	nfft/1e6/FFTyz.Elapsed()*2.5*log(ngrid3)/log(2)/1e3*2.0/3.0); 
    fprintf(fp, "#      FFTx time:   %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n", 
        FFTx.Elapsed(), 
    	nfft/1e6/FFTx.Elapsed()/3.0, 
    	nfft/1e9/FFTx.Elapsed()*6.0*sizeof(Float)/3.0,
    	nfft/1e6/FFTx.Elapsed()*2.5*log(ngrid3)/log(2)/1e3/3.0); 
#endif
    // Approximating number of Ylm cells as FFT/2.  
    // Each stores one float, but nearly all load one float too.
    fprintf(fp, "#       Ylm time:   %8.4f s, %6.3f GB/s\n", YlmTime.Elapsed(),
	(nfft-ngrid3)/2.0/1e9/YlmTime.Elapsed()*sizeof(Float)*2.0
    	);
    fprintf(fp, "#      Hist time:   %8.4f s\n", Hist.Elapsed());
    fprintf(fp, "#   Extract time:   %8.4f s\n", Extract.Elapsed());
    // We're doing two FFTs per loop and then one extra, so like 2*N+1
    // Hence N examples of A*Bt, each of which is 3 Floats of load/store
    fprintf(fp, "#      A*Bt time:   %8.4f s, %6.3f M/s of A=A*Bt, %6.3f GB/s\n", AtimesB.Elapsed(), 
    		(nfft/2.0/ngrid3-0.5)*ngrid3/1e6/AtimesB.Elapsed(), 
    		(nfft/2.0/ngrid3-0.5)*ngrid3/1e9/AtimesB.Elapsed()*3.0*sizeof(Float) );
    fprintf(fp, "# Total time:       %8.4f s\n", Total.Elapsed());
    if (Misc.Elapsed()>0.0) {
	fprintf(fp, "#\n# Misc time:          %8.4f s\n", Misc.Elapsed());
    }
    return;
}

class ThreadCount {
    int *cnt;
    int max;
  public:
    ThreadCount(int max_threads) {
	max = max_threads;
	int err=posix_memalign((void **) &cnt, PAGE, sizeof(int)*8*max); assert(err==0);
	for (int j=0; j<max*8; j++) cnt[j]=0;
	return;
    }
    ~ThreadCount() { free(cnt); return; }
    void add() {
	cnt[omp_get_thread_num()*8]++;
    }
    void print(FILE *fp) {
	for (int j=0; j<max; j++) if (cnt[j*8]>0) 
	    fprintf(fp, "# Thread %2d = %d\n", j, cnt[j*8]);
    }
};

#define MAX_THREADS 128     
ThreadCount Ylm_count(MAX_THREADS);

void usage() {
    fprintf(stderr, "FFTCORR: Error in command-line \n");
    fprintf(stderr, "   -n <int> (or -ngrid): FFT linear grid size for a cubic box\n");
    fprintf(stderr, "   -n3 <int> <int> <int> (or -ngrid3): FFT linear grid sizes for rectangle\n");
    fprintf(stderr, "             -n3 will outrank -n\n");
    fprintf(stderr, "   -ell <int> (or -maxell): Multipole to compute.\n");
    fprintf(stderr, "   -b <float> (or -box): Bounding box size.  Must exceed value in input file.\n");
    
    fprintf(stderr, "             <0 will default to value in input file.\n");
    fprintf(stderr, "   -r <float> (or -sep): Max separation.  Cannot exceed value in input file.\n");
    fprintf(stderr, "             <0 will default to value in input file.\n");
    fprintf(stderr, "   -dr <float> (or -dsep): Binning of separation.\n");
    fprintf(stderr, "   -kmax <float>: Maximum wavenumber for power spectrum.\n");
    fprintf(stderr, "   -dk <float>: Binning of wavenumber.\n");
    fprintf(stderr, "   -periodic (or -p): Configure for cubic periodic box.\n");
    fprintf(stderr, "   -zeromean (or -z): Configure for cubic periodic box and set mean density to zero.\n");
    fprintf(stderr, "   -in <filename>:  Input file name\n");
    fprintf(stderr, "   -in2 <filename>: Second input file name\n");
    fprintf(stderr, "   -out <filename>: Output file name, default to stdout\n");
    fprintf(stderr, "   -exp <int> (or -e): Use the wide-angle exponent of Slepian & Eisenstein 2016 (niche users only)\n");
    fprintf(stderr, "\n");
    exit(1);
}

/* =========================================================================== */

int main(int argc, char *argv[]) {
    // Need to get this information.
    Total.Start();
    // Here are some defaults
    Float sep = -123.0;  // Default to max_sep from the file
    Float dsep = 10.0;
    Float kmax = 0.03;
    Float dk = 0.01;
    int maxell = 4;   
    int wide_angle_exponent = 0;   
    int ngridCube = 256;
    int qperiodic = 0; 
    int ngrid[3] = { -1, -1, -1};
    Float cell = -123.0;   // Default to what's implied by the file
    const char default_fname[] = "/tmp/corrRR.dat";
    char *infile = NULL;
    char *infile2 = NULL;
    char *outfile = NULL;

    int i=1;
    while (i<argc) {
             if (!strcmp(argv[i],"-ngrid")||!strcmp(argv[i],"-n")) ngridCube = atoi(argv[++i]);
	else if (!strcmp(argv[i],"-ngrid3")||!strcmp(argv[i],"-n3")) {
		ngrid[0] = atoi(argv[++i]); ngrid[1] = atoi(argv[++i]); ngrid[2] = atoi(argv[++i]);
	}
	else if (!strcmp(argv[i],"-maxell")||!strcmp(argv[i],"-ell")) maxell = atoi(argv[++i]);
	else if (!strcmp(argv[i],"-exp")||!strcmp(argv[i],"-e")) wide_angle_exponent = atoi(argv[++i]);
	else if (!strcmp(argv[i],"-sep")||!strcmp(argv[i],"-r")) sep = atof(argv[++i]);
	else if (!strcmp(argv[i],"-dsep")||!strcmp(argv[i],"-dr")) dsep = atof(argv[++i]);
	else if (!strcmp(argv[i],"-kmax")||!strcmp(argv[i],"-k")) kmax = atof(argv[++i]);
	else if (!strcmp(argv[i],"-dk")||!strcmp(argv[i],"-dk")) dk = atof(argv[++i]);
	else if (!strcmp(argv[i],"-cell")||!strcmp(argv[i],"-c")) cell = atof(argv[++i]);
	else if (!strcmp(argv[i],"-in")||!strcmp(argv[i],"-i")) infile = argv[++i];
	else if (!strcmp(argv[i],"-in2")||!strcmp(argv[i],"-i2")) infile2 = argv[++i];
	else if (!strcmp(argv[i],"-out")||!strcmp(argv[i],"-o")) outfile = argv[++i];
	else if (!strcmp(argv[i],"-periodic")||!strcmp(argv[i],"-p")) qperiodic = 1;
	else if (!strcmp(argv[i],"-zeromean")||!strcmp(argv[i],"-z")) qperiodic = 2;
	else usage();
	i++;
    }

    assert(ngrid>0);
    assert(maxell>=0 && maxell%2==0);
    assert(wide_angle_exponent%2==0);  // Must be an even number
    assert(sep!=0.0);
    assert(dsep>0.0);
    assert(kmax!=0.0);
    assert(dk>0.0);
    assert(qperiodic==0||sep>0);  // If qperiodic is set, user must supply a sep
    if (infile==NULL) infile = (char *)default_fname;
    if (outfile!=NULL) { FILE *discard=freopen(outfile,"w", stdout); 
    		assert(discard!=NULL&&stdout!=NULL); }

    if (ngrid[0]<=0) ngrid[0] = ngrid[1] = ngrid[2] = ngridCube;
    assert(ngrid[0]>0);
    assert(ngrid[1]>0);
    assert(ngrid[2]>0);

    #ifdef OPENMP
	fprintf(stdout,"# Running with %d threads\n", omp_get_max_threads());
    #else
	fprintf(stdout,"# Running single threaded.\n");
    #endif

    setup_wavelet();
    Grid g(infile, ngrid, cell, sep, qperiodic);
    fprintf(stdout, "# Using wide-angle exponent %d\n", wide_angle_exponent);
    g.read_galaxies(infile, infile2, qperiodic);
    // The input grid is now in g.dens
    sep = g.setup_corr(sep, kmax);
    Histogram h(maxell, sep, dsep);
    Histogram kh(maxell, kmax, dk);
    g.correlate(maxell, h, kh, wide_angle_exponent);

    Ylm_count.print(stdout);
    fprintf(stdout, "# Anisotropic power spectrum:\n");
    kh.print(stdout,1);
    fprintf(stdout, "# Anisotropic correlations:\n");
    h.print(stdout,0);
    // We want to use the correlation at zero lag as the I normalization
    // factor in the FKP power spectrum.
    fprintf(stdout, "#\n# Zero-lag correlations are %14.7e\n", h.zerolag);
    // Integral of power spectrum needs a d^3k/(2 pi)^3, which is (1/L)^3 = (1/(cell_size*ngrid))^3
    fprintf(stdout, "#\n# Integral of power spectrum is %14.7e\n", 
    	kh.sum()/(g.cell_size*g.cell_size*g.cell_size*ngrid[0]*ngrid[1]*ngrid[2]));

    Total.Stop();
    uint64 nfft=1; 
    for (int j=0; j<=maxell; j+=2) nfft+=2*(2*j+1);
    nfft*=g.ngrid3;
    fprintf(stdout,"#\n");
    ReportTimes(stdout, nfft, g.ngrid3, g.cnt);
    return 0;
}



/* ============== Spherical Harmonic routine ============== */


void makeYlm(Float *Ylm, int ell, int m, int n[3], int n1, Float *xcell, Float *ycell, Float *z, Float *dens, int exponent) {
    // We're not actually returning Ylm here.
    // m>0 will return Re(Y_lm)*sqrt(2)
    // m<0 will return Im(Y_l|m|)*sqrt(2)
    // m=0 will return Y_l0
    // These are not including common minus signs, since we're only
    // using them in matched squares.
    //
    // Input x[n[0]], y[n[1]], z[n[2]] are the x,y,z centers of this row of bins
    // n1 is supplied so that Ylm[n[0]][n[1]][n1] can be handled flexibly
    //
    // If dens!=NULL, then it should point to a [n[0]][n[1]][n1] vector that will be multiplied
    // element-wise onto the results.  This can save a store/load to main memory.
    // 
    // If exponent!=0, then we will attach a dependence of r^exponent to the Ylm's
    // exponent must be an even number
    assert(exponent%2==0);
    YlmTime.Start();
    Float isqpi = sqrt(1.0/M_PI);
    if (m!=0) isqpi *= sqrt(2.0);    // Do this up-front, so we don't forget
    Float tiny = 1e-20;

    const uint64 nc3 = (uint64)n[0]*n[1]*n1;
    if (ell==0&&m==0&&exponent==0) {
	// This case is so easy that we'll do it first and skip the rest of the set up
	if (dens==NULL) set_matrix(Ylm, 1.0/sqrt(4.0*M_PI), nc3, n[0]);
	          else copy_matrix(Ylm, dens, 1.0/sqrt(4.0*M_PI), nc3, n[0]);
	YlmTime.Stop();
	return;
    }

    const int cn2 = n[2];    // To help with loop vectorization
    Float *z2, *z3, *z4, *ones;
    int err=posix_memalign((void **) &z2, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
    err=posix_memalign((void **) &z3, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
    err=posix_memalign((void **) &z4, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
    err=posix_memalign((void **) &ones, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
    for (int k=0; k<cn2; k++) {
	z2[k] = z[k]*z[k];
	z3[k] = z2[k]*z[k];
	z4[k] = z3[k]*z[k];
	ones[k] = 1.0;
    }

    Ylm[0] = -123456.0;    // A sentinal value

    #pragma omp parallel for YLM_SCHEDULE
    for (uint64 i=0; i<n[0]; i++) {
    	Ylm_count.add();
	Float *ir2;    // Need some internal workspace
	err=posix_memalign((void **) &ir2, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
	Float *rpow;
	err=posix_memalign((void **) &rpow, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
        Float x = xcell[i], x2 = x*x;
	Float *Y = Ylm+i*n[1]*n1;
	Float *D = dens+i*n[1]*n1;
	Float *R;
	for (int j=0; j<n[1]; j++, Y+=n1, D+=n1) {
	    if (dens==NULL) D=ones;
	    Float y = ycell[j], y2 = y*y, y3 = y2*y, y4 = y3*y;
	    for (int k=0; k<cn2; k++) ir2[k] = 1.0/(x2+y2+z2[k]+tiny);
	    // Now figure out the exponent r^n
	    if (exponent==0) R = ones;
	    else if (exponent>0) {
		// Fill R with r^exponent
	        R = rpow;
		for (int k=0; k<cn2; k++) R[k] = (x2+y2+z2[k]);
		for (int e=exponent; e>2; e-=2) 
		    for (int k=0; k<cn2; k++) R[k] *= (x2+y2+z2[k]);
	    } else {
		// Fill R with r^exponent
	        R = rpow;
		for (int k=0; k<cn2; k++) R[k] = ir2[k];
		for (int e=exponent; e<-2; e+=2) 
		    for (int k=0; k<cn2; k++) R[k] *= ir2[k];
	    }
	    // Now ready to compute
	    if (ell==2) {
	        if (m==2) 
		    for (int k=0; k<cn2; k++) 
			Y[k] = D[k]*R[k]*isqpi*sqrt(15./32.)*(x2-y2)*ir2[k];
		else if (m==1) 
		    for (int k=0; k<cn2; k++) 
			Y[k] = D[k]*R[k]*isqpi*sqrt(15./8.)*x*z[k]*ir2[k];
		else if (m==0) 
		    for (int k=0; k<cn2; k++) 
			Y[k] = D[k]*R[k]*isqpi*sqrt(5./16.)*(2.0*z2[k]-x2-y2)*ir2[k];
		else if (m==-1) 
		    for (int k=0; k<cn2; k++) 
			Y[k] = D[k]*R[k]*isqpi*sqrt(15./8.)*y*z[k]*ir2[k];
		else if (m==-2) 
		    for (int k=0; k<cn2; k++) 
			Y[k] = D[k]*R[k]*isqpi*sqrt(15./32.)*2.0*x*y*ir2[k];
	    }
	    else if (ell==4) {
	        if (m==4) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/16.0*sqrt(35./2.)*(x2*x2-6.0*x2*y2+y4)*ir2[k]*ir2[k]; 
		else if (m==3) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/8.0*sqrt(35.)*(x2-3.0*y2)*z[k]*x*ir2[k]*ir2[k]; 
		else if (m==2) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/8.0*sqrt(5./2.)*(6.0*z2[k]*(x2-y2)-x2*x2+y4)*ir2[k]*ir2[k]; 
		else if (m==1) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/8.0*sqrt(5.)*3.0*(4.0/3.0*z2[k]-x2-y2)*x*z[k]*ir2[k]*ir2[k]; 
		else if (m==0) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/16.0*8.0*(z4[k]-3.0*z2[k]*(x2+y2)+3.0/8.0*(x2*x2+2.0*x2*y2+y4))*ir2[k]*ir2[k];
		else if (m==-1) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/8.0*sqrt(5.)*3.0*(4.0/3.0*z2[k]-x2-y2)*y*z[k]*ir2[k]*ir2[k]; 
		else if (m==-2) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/8.0*sqrt(5./2.)*(2.0*x*y*(6.0*z2[k]-x2-y2))*ir2[k]*ir2[k]; 
		else if (m==-3) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/8.0*sqrt(35.)*(3.0*x2*y-y3)*z[k]*ir2[k]*ir2[k]; 
		else if (m==-4) 
		    for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]*isqpi*3.0/16.0*sqrt(35./2.)*(4.0*x*(x2*y-y3))*ir2[k]*ir2[k]; 
	    } else if (ell==0) {
	        // We only get here if exponent!=0
		for (int k=0; k<cn2; k++) 
		    Y[k] = D[k]*R[k]/sqrt(4.0*M_PI);
	    }
	}
	free(ir2);
    }
    assert(Ylm[0]!=123456.0);  // This traps whether the user entered an illegal (ell,m)
    free(z2);
    free(z3);
    free(z4);
    YlmTime.Stop();
    return;
}
