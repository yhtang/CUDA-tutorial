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

// host code entrance
int main( int argc, char **argv )
{
    // the grayscale image is stored in image.data(), with one value for each pixel
    png_image image( argc > 1 ? argv[1] : "original.png" );

    // the gaussian filter
    const static int r = 3;
    const static int d = 7;
    const static float gauss_filter [] = {
        0.0049, 0.0092, 0.0134, 0.0152, 0.0134, 0.0092, 0.0049, \
        0.0092, 0.0172, 0.0250, 0.0283, 0.0250, 0.0172, 0.0092, \
        0.0134, 0.0250, 0.0364, 0.0412, 0.0364, 0.0250, 0.0134, \
        0.0152, 0.0283, 0.0412, 0.0467, 0.0412, 0.0283, 0.0152, \
        0.0134, 0.0250, 0.0364, 0.0412, 0.0364, 0.0250, 0.0134, \
        0.0092, 0.0172, 0.0250, 0.0283, 0.0250, 0.0172, 0.0092, \
        0.0049, 0.0092, 0.0134, 0.0152, 0.0134, 0.0092, 0.0049
    };

    png_image new_image = image;

    // image.data() returns pointer to the pixel data

    #pragma omp parallel for
    for( int i = 0; i < image.height; i++ ) {
        for( int j = 0; j < image.width; j++ ) {
            float sum = 0.f;
            for( int k = 0; k < d; k++ ) {
                for( int l = 0; l < d; l++ ) {
                    int K = i - r + k;
                    int L = j - r + l;
                    // using 'symmetric' boundary here
                    if( K < 0 ) K = -K;
                    if( L < 0 ) L = -L;
                    if( K >= image.height ) K = 2 * image.height - K;
                    if( L >= image.width ) L = 2 * image.width - L;
                    // adding the contributions
                    sum += image.pixel[K * image.width + L] * gauss_filter[k * d + l];
                }
            }
            new_image.pixel[i * new_image.width + j] = sum;
        }
    }

    new_image.save( "result.png" );
}
