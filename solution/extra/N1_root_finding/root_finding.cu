#include <cstdio>
#include <iostream>
#include <vector>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../util/util.h"

#if 0
__device__ double f( double x )
{
    return 36.640625 + x * ( -96.060546875 + x * ( 64.01513671875 + x * ( -14.572509765625 + x ) ) );
}
#endif

#if 1
__device__ double f( double x )
{
    return 23.0706125743761 + x * ( -48.9931636222408 + x * ( 34.6514797153910 + x * ( -9.9612755239323 + x ) ) );
}
#endif

__global__ void find_root( const double left, const double right, const double tolerance, const int n )
{
    if( blockIdx.x * blockDim.x + threadIdx.x == 0 ) printf( "searching within [%.15lf,%.15lf)\n", left, right );
    __syncthreads();

    double delta = ( right - left ) / n;
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        double a = i * delta + left;
        double b = a + delta;
        if( signbit( f(a) * f(b) ) ) {
            if( delta < tolerance ) {
                printf( "root found as %.15lf\n", ( b + a ) * 0.5 );
            } else {
                find_root <<< 1, 1024>>>( a, b, tolerance, n );
                cudaDeviceSynchronize();
            }
        }
    }
}

// host code entrance
int main( int argc, char **argv )
{
    double tolerance = argc > 1 ? atof( argv[1] ) : 1E-7;
    int n = argc > 2 ? atof( argv[2] ) : 1024;

    find_root <<< 1, 1024>>>( 0.0, 1.0, tolerance, n );
    cudaDeviceSynchronize();
}
