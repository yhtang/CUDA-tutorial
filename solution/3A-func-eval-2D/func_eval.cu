#include <cstdio>
#include <iostream>
#include <vector>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../util/util.h"

__inline__ __host__ __device__ double f( double x, double y )
{
    return sin( 5.0 * x ) * cos( 16.0 * y ) * exp( x );
}

__global__ void evaluate( double *z, const int n )
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    z[i + j * n] = f( ( double )i / ( double )n, ( double )j / ( double )n );
}

// host code entrance
int main( int argc, char **argv )
{
    int N = 4 * 1024;

    // timing register
    double t_CPU_0, t_CPU_1, t_GPU_0, t_GPU_1;
    // allocate host memory
    double *hst_z, *ref_z;
    hst_z = new double[N * N];
    ref_z = new double[N * N];
    // allocate device memory
    double *dev_z;
    cudaMalloc( &dev_z, N * N * sizeof( double ) );

    t_GPU_0 = get_time();

    // do computation on GPU
    dim3 n_blocks( N / 32, N / 32, 1 ), n_threads( 32, 32, 1 );

    evaluate <<< n_blocks, n_threads >>> ( dev_z, N );
    // copy result back to CPU
    cudaMemcpy( hst_z, dev_z, N * N * sizeof( double ), cudaMemcpyDefault );

    t_GPU_1 = get_time();

    t_CPU_0 = get_time();

    // calculate reference value
    for( int i = 0; i < N; i++ ) {
        for( int j = 0 ; j < N ; j++ ) {
            ref_z[i + j * N] = f( ( double )i / ( double )N, ( double )j / ( double )N );
        }
    }

    t_CPU_1 = get_time();

    // compare
    bool match = true;
    for( int i = 0; i < N; i++ ) {
        for( int j = 0 ; j < N ; j++ ) {
            match = match &&
                    ( fabs( ref_z[i] - hst_z[i] ) < 8 * std::numeric_limits<double>::epsilon() );
        }
    }

    // output
    std::cout << "Computation on CPU took " << t_CPU_1 - t_CPU_0 << " secs." << std::endl;
    std::cout << "Computation on GPU took " << t_GPU_1 - t_GPU_0 << " secs." << std::endl;
    std::cout << "CPU/GPU result match: " << ( match ? "YES" : "NO" ) << std::endl;

    // free up resources
    delete [] hst_z;
    delete [] ref_z;
    cudaDeviceReset();
}
