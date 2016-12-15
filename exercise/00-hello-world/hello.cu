#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hello_gpu()
{
    printf( "\"Hello, world!\", says the GPU.\n" );
}

void hello_cpu()
{
    printf( "\"Hello, world!\", says the CPU.\n" );
}

// host code entrance
int main( int argc, char **argv )
{
    hello_cpu();
    hello_gpu <<< 16, 512>>>();
    cudaDeviceSynchronize();
}
