#include <cstdio>
#include <vector>
#include <cstdlib>
#include <omp.h>

struct Timer {
	double t0;
	Timer() : t0( omp_get_wtime() ) {}
	double stop() { return omp_get_wtime() - t0; }
}; 

__global__ void daxpy( double *x, double *y, double *z, const int N ) {
	int n_threads = gridDim.x * blockDim.x;
	int my_tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = my_tid; i < N; i += n_threads ) z[i] = 2. * x[i] + y[i];	
}

int main() {
	int N = 128 * 1024 * 1024;
	std::vector<double> x(N), y(N), z(N);

	// initialize the array with random contents
	printf("initializing...");
	fflush(stdout);
	for(int i = 0; i < N; i++) {
		x[i] = rand() / double(RAND_MAX);
		y[i] = rand() / double(RAND_MAX);
	}
	printf("done\n");

	// calculate on CPU, serial
	Timer t_serial;
	for(int i=0;i<N;i++) z[i] = 2. * x[i] + y[i];
	printf("CPU, serial: %.3lf s\n", t_serial.stop() );

	// calculate on CPU, parallel
	Timer t_parallel;
	#pragma omp parallel for
	for(int i=0;i<N;i++) z[i] = 2. * x[i] + y[i];
	printf("CPU, %d threads: %.3lf s\n", omp_get_max_threads(), t_parallel.stop() );

	// GPU
	double *dev_x, *dev_y, *dev_z;
	std::vector<double> hst_z(N);
	cudaMalloc( &dev_x, N * sizeof(double) );
	cudaMalloc( &dev_y, N * sizeof(double) );
	cudaMalloc( &dev_z, N * sizeof(double) );
	Timer t_upload;
	cudaMemcpy( dev_x, x.data(), N * sizeof(double), cudaMemcpyDefault );
	cudaMemcpy( dev_y, y.data(), N * sizeof(double), cudaMemcpyDefault );
	printf("GPU, data upload: %.3lf s\n", t_upload.stop() );
	Timer t_gpu;
	daxpy<<< 30, 1024 >>>( dev_x, dev_y, dev_z, N );
	cudaDeviceSynchronize();
	printf("GPU, computation: %.3lf s\n", t_gpu.stop() );
        Timer t_download;
        cudaMemcpy( hst_z.data(), dev_z, N * sizeof(double), cudaMemcpyDefault );
        printf("GPU, data download: %.3lf s\n", t_download.stop() );

	// correctness check
	bool pass = true;
	for(int i=0;i<N;i++) if ( std::fabs( z[i] - hst_z[i] ) > 1e-10 ) pass = false;
	printf("Correctness check: %s\n", pass ? "PASS" : "FAIL" );

}

