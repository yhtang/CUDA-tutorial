#include <cstdio>
#include <iostream>
#include <vector>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ __inline__ long long __shfl( long long var, int srcLane, int width = warpSize )
{
    int lo, hi;
    asm volatile( "mov.b64 {%0, %1}, %2; " : "=r"( lo ), "=r"( hi ) : "l"( var ) );
    lo = __shfl( lo, srcLane, width );
    hi = __shfl( hi, srcLane, width );
    long long r;
    asm volatile( "mov.b64 %0, {%1, %2}; " : "=l"( r ) : "r"( lo ), "r"( hi ) );
    return r;
}

__device__ __inline__ double __shfl( double var, int srcLane, int width = warpSize )
{
    int lo, hi;
    asm volatile( "mov.b64 {%0, %1}, %2; " : "=r"( lo ), "=r"( hi ) : "d"( var ) );
    lo = __shfl( lo, srcLane, width );
    hi = __shfl( hi, srcLane, width );
    double r;
    asm volatile( "mov.b64 %0, {%1, %2}; " : "=d"( r ) : "r"( lo ), "r"( hi ) );
    return r;
}

__global__ void shuffle_int64_test()
{
    long long i = threadIdx.x;
    long long p = __shfl( i, warpSize - 1 - threadIdx.x );
    printf( "thread %d got value %ld\n", threadIdx.x, p );
}

__global__ void shuffle_float64_test()
{
    double i = threadIdx.x;
    double p = __shfl( i, warpSize - 1 - threadIdx.x );
    printf( "thread %d got value %lf\n", threadIdx.x, p );
}

int main()
{
    shuffle_int64_test <<< 1, 32>>>();
    cudaDeviceSynchronize();
    shuffle_float64_test <<< 1, 32>>>();
    cudaDeviceSynchronize();
}
