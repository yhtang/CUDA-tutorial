#include <cstdio>
#include <iostream>
#include <vector>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../util/util.h"
#include "../util/math.h"

template<typename REAL>
__inline__ __host__ __device__ REAL f( REAL x )
{
#ifdef __CUDA_ARCH__
    return cuda::sin( REAL( 2.0 ) * x ) * cuda::cos( REAL( 7.0 ) * x ) * cuda::exp( x );
#else
    return std::sin( REAL( 2.0 ) * x ) * std::cos( REAL( 7.0 ) * x ) * std::exp( x );
#endif
}

template<typename REAL>
__global__ void evaluate( REAL *y, const int n )
{
    int i = global_thread_id();
    y[i] = f( ( REAL )i / ( REAL )n );
}

// host code entrance
template<typename REAL>
void evaluate()
{
    int N = 128 * 1024 * 1024;

    // timing register
    double t_CPU_0, t_CPU_1, t_GPU_0, t_GPU_1, t_GPU_2;
    // allocate host memory
    REAL *hst_y, *ref_y;
    hst_y = new REAL[N];
    ref_y = new REAL[N];
    // allocate device memory
    REAL *dev_y;
    cudaMalloc( &dev_y, N * sizeof( REAL ) );

    t_GPU_0 = get_time();

    // do computation on GPU
    evaluate <<< N / 1024, 1024 >>> ( dev_y, N );
    cudaDeviceSynchronize();

    t_GPU_1 = get_time();

    // copy result back to CPU
    cudaMemcpy( hst_y, dev_y, N * sizeof( REAL ), cudaMemcpyDefault );

    t_GPU_2 = get_time();

    t_CPU_0 = get_time();

    // calculate reference value
    #pragma omp parallel for
    for( int i = 0; i < N; i++ ) ref_y[i] = f( ( REAL )i / ( REAL )N );

    t_CPU_1 = get_time();

    // compare
    bool match = true;
    for( int i = 0; i < N; i++ ) {
        match = match &&
                ( fabs( ref_y[i] - hst_y[i] ) < 8 * std::numeric_limits<REAL>::epsilon() );
    }

    // output
    std::cout << "Computation on CPU took " << t_CPU_1 - t_CPU_0 << " secs." << std::endl;
    std::cout << "Computation on GPU took " << t_GPU_1 - t_GPU_0 << " secs." << std::endl;
    std::cout << "Data transfer from GPU took " << t_GPU_2 - t_GPU_1 << " secs." << std::endl;
    std::cout << "CPU/GPU result match: " << ( match ? "YES" : "NO" ) << std::endl;
    std::cout << "Precision mode: " << ( sizeof( REAL ) == 4 ? "single" : "double" ) << std::endl;

    // free up resources
    delete [] hst_y;
    delete [] ref_y;
    cudaFree( dev_y );
}

int main()
{
    evaluate<double>();
    evaluate<float>();
}
