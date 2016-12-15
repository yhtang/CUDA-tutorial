#include <cstdio>
#include <iostream>
#include <vector>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../util/util.h"

__inline__ __host__ __device__ double f( double x )
{
    return sin( 2.0 * x ) * cos( 7.0 * x ) * exp( x );
}

__global__ void evaluate( double *y, const int n )
{
    int i = global_thread_id();
    y[i] = f( ( double )i / ( double )n );
}

// host code entrance
int main( int argc, char **argv )
{
    int N = 128 * 1024 * 1024;

    // timing register
    double t_CPU_0, t_CPU_1;
    // allocate host memory
    double *hst_y, *ref_y;
    hst_y = new double[N];
    ref_y = new double[N];
    // allocate device memory
    double *dev_y;
    cudaMalloc( &dev_y, N * sizeof( double ) );

    // TO-DO: insert event 0 here
    insert_zeroth_event;

    // do computation on GPU
    evaluate <<< N / 1024, 1024 >>> ( dev_y, N );

    // TO-DO: insert event 1 here
    insert_first_event;

    // copy result back to CPU
    cudaMemcpyAsync( hst_y, dev_y, N * sizeof( double ), cudaMemcpyDefault );

    // TO-DO: insert event 2 here
    insert_second_event;

    cudaDeviceSynchronize();

    t_CPU_0 = get_time();

    // calculate reference value
    #pragma omp parallel for
    for( int i = 0; i < N; i++ ) ref_y[i] = f( ( double )i / ( double )N );

    t_CPU_1 = get_time();

    // compare
    bool match = true;
    for( int i = 0; i < N; i++ ) {
        match = match &&
                ( fabs( ref_y[i] - hst_y[i] ) < 8 * std::numeric_limits<double>::epsilon() );
    }

    // output
    // TO-DO query time elapsed between events
    std::cout << "Computation on CPU took " << t_CPU_1 - t_CPU_0 << " secs." << std::endl;
    std::cout << "Computation on GPU took " << get_time_between_event_0_and_1 << " secs." << std::endl;
    std::cout << "Data transfer from GPU took " << get_time_between_event_1_and_2 << " secs." << std::endl;
    std::cout << "CPU/GPU result match: " << ( match ? "YES" : "NO" ) << std::endl;

    // free up resources
    delete [] hst_y;
    delete [] ref_y;
    cudaDeviceReset();
}
