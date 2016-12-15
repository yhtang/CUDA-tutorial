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
    // TO-DO
}

// host code entrance
int main( int argc, char **argv )
{
    int N = 128 * 1024 * 1024;

    // allocate host memory
    float hst_sum = 0.f, ref_sum = 0.f;
    float *hst_value;
    hst_value = new float[N];

    srand( 0 ); // deterministic result can be helpful for debugging
    for( int i = 0; i < N; i++ ) hst_value[i] = float( rand() ) / float( RAND_MAX );

    // allocate device memory
    /*TO-DO*/

    // do computation on GPU
    mean <<< /*TO-DO*/ >>> ( /*TO-DO*/ );

    // copy result back to CPU
    /*TO-DO*/

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
