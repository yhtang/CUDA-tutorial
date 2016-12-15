#include <cstdio>
#include <iostream>
#include <vector>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../util/util.h"

__global__ void mean( float *value, float *sum, const int n )
{
    float my_sum = 0;
    for( int i = global_thread_id(); i < n; i += global_thread_num() ) {
        my_sum += value[i];
    }
    atomicAdd( sum, my_sum );
}

// host code entrance
int main( int argc, char **argv )
{
    int N = 128 * 1024 * 1024;

    // allocate host memory
    float hst_sum = 0.f, ref_sum = 0.f;
    float *hst_value;
    hst_value = new float[N];
    // allocate device memory
    float *dev_value, *dev_sum;
    cudaMalloc( &dev_value, N * sizeof( float ) );
    cudaMalloc( &dev_sum, sizeof( float ) );

    srand( 0 ); // deterministic result can be helpful for debugging
    for( int i = 0; i < N; i++ ) hst_value[i] = float( rand() ) / float( RAND_MAX );

    cudaMemset( dev_sum, 0, sizeof( float ) );
    cudaMemcpy( dev_value, hst_value, N * sizeof( float ), cudaMemcpyDefault );

    // do computation on GPU
    mean <<< 16, 1024 >>> ( dev_value, dev_sum, N );

    // copy result back to CPU
    cudaMemcpy( &hst_sum, dev_sum, sizeof( float ), cudaMemcpyDefault );

    // calculate reference value
    for( int i = 0; i < N; i++ ) ref_sum += hst_value[i];

    hst_sum /= N;
    ref_sum /= N;

    // compare & output
    std::cout << "CPU result: " << ref_sum << std::endl;
    std::cout << "GPU result: " << hst_sum << std::endl;
    std::cout << "CPU/GPU result match: " << ( fabs( hst_sum - ref_sum ) < 10 * std::numeric_limits<float>::epsilon() ? "YES" : "NO" ) << std::endl;

    // free up resources
    delete [] hst_value;
    cudaDeviceReset();
}
