#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <limits>
#include <csignal>

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../util/util.h"
#include "../util/png.h"

struct png_image {
    uint width, height;
    std::vector<unsigned char> pixel;

    png_image( const char fn[] ) {
        load( fn );
    }
    uint load( const char fn[] ) {
        return lodepng::decode( pixel, width, height, fn, LCT_GREY, 8 );
    }
    uint save( const char fn[] ) {
        return lodepng::encode( fn, &pixel[0], width, height, LCT_GREY, 8 );
    }
    unsigned char *data() {
        return &pixel[0];
    }
};

__global__ void filter( const unsigned char * image, unsigned char * new_image, const int width, const int height )
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if( i < height && j < width ) {
        float sum = 0.f;
        for( int k = 0; k < 3; k++ ) {
            for( int l = 0; l < 3; l++ ) {
                int K = i - 1 + k;
                int L = j - 1 + l;
                // using 'symmetric' boundary here
                if( K < 0 ) K = -K;
                if( L < 0 ) L = -L;
                if( K >= height ) K = 2 * height - K;
                if( L >= width )  L = 2 * width - L;
                // adding the contributions
                sum += image[K * width + L] * 0.111111;
            }
        }
        new_image[i * width + j] = sum;
    }
}

// host code entrance
int main( int argc, char **argv )
{
    // the grayscale image is stored in image.data(), with one value for each pixel
    png_image image( argc > 1 ? argv[1] : "original.png" );
    png_image new_image = image;

    unsigned char *dev_image, *dev_new_image;
    cudaMalloc( &dev_image, sizeof( unsigned char ) * image.width * image.height );
    cudaMalloc( &dev_new_image, sizeof( unsigned char ) * image.width * image.height );
    cudaMemcpy( dev_image, image.data(), sizeof( unsigned char ) * image.width * image.height, cudaMemcpyDefault );
    
    // image.data() returns pointer to the pixel data
    dim3 n_threads( 16, 16 );
    dim3 n_blocks( ( image.width + n_threads.x - 1 ) / n_threads.x, ( image.height + n_threads.y - 1 ) / n_threads.y );
    filter <<< n_blocks, n_threads>>>( dev_image, dev_new_image, image.width, image.height );

    cudaMemcpy( new_image.data(), dev_new_image, sizeof( unsigned char ) * image.width * image.height, cudaMemcpyDefault );

    new_image.save( "result.png" );
}
