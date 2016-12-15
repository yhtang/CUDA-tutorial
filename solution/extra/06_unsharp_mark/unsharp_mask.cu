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

float point_filter( png_image &image, const float filter[], int i, int j, int d )
{
    float sum = 0.f;
    int r = ( d - 1 ) / 2;
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
            sum += image.pixel[K * image.width + L] * filter[k * d + l];
        }
    }
    return sum;
}

// host code entrance
int main( int argc, char **argv )
{
    // the grayscale image is stored in image.data(), with one value for each pixel
    png_image image( argc > 1 ? argv[1] : "original.png" );

    const static float threshold = 125.0;
    // filters
    const static float gauss_filter [] = {
        0.0049, 0.0092, 0.0134, 0.0152, 0.0134, 0.0092, 0.0049, \
        0.0092, 0.0172, 0.0250, 0.0283, 0.0250, 0.0172, 0.0092, \
        0.0134, 0.0250, 0.0364, 0.0412, 0.0364, 0.0250, 0.0134, \
        0.0152, 0.0283, 0.0412, 0.0467, 0.0412, 0.0283, 0.0152, \
        0.0134, 0.0250, 0.0364, 0.0412, 0.0364, 0.0250, 0.0134, \
        0.0092, 0.0172, 0.0250, 0.0283, 0.0250, 0.0172, 0.0092, \
        0.0049, 0.0092, 0.0134, 0.0152, 0.0134, 0.0092, 0.0049
    };
    const static float sharp_filter [] = {
        -0.0015, -0.0081, -0.0191, -0.0243, -0.0191, -0.0081, -0.0015, \
        -0.0081, -0.0298, -0.0382, -0.0234, -0.0382, -0.0298, -0.0081, \
        -0.0191, -0.0382, 0.0523, 0.1792, 0.0523, -0.0382, -0.0191, \
        -0.0243, -0.0234, 0.1792, 0.4136, 0.1792, -0.0234, -0.0243, \
        -0.0191, -0.0382, 0.0523, 0.1792, 0.0523, -0.0382, -0.0191, \
        -0.0081, -0.0298, -0.0382, -0.0234, -0.0382, -0.0298, -0.0081, \
        -0.0015, -0.0081, -0.0191, -0.0243, -0.0191, -0.0081, -0.0015,
    };
    const static float dx_filter [] = {
        -1.0f, 0.0f, 1.0f, \
        -1.0f, 0.0f, 1.0f, \
        -1.0f, 0.0f, 1.0f
    };
    const static float dy_filter [] = {
        -1.0f, -1.0f, -1.0f, \
        0.0f,  0.0f,  0.0f, \
        1.0f,  1.0f,  1.0f
    };

    png_image new_image = image;

    // image.data() returns pointer to the pixel data

    #pragma omp parallel for
    for( int i = 0; i < image.height; i++ ) {
        for( int j = 0; j < image.width; j++ ) {
            // calculate local contrast
            float dx = point_filter( image, dx_filter, i, j, 3 );
            float dy = point_filter( image, dy_filter, i, j, 3 );
            float contrast = std::sqrt( dx * dx + dy * dy );
            // sharpen the edges and blur everything else
            if( contrast > threshold ) {
                new_image.pixel[i * new_image.width + j] = min( max( 0.0f, point_filter( image, sharp_filter, i, j, 7 ) ), 255.0 );
            } else {
                new_image.pixel[i * new_image.width + j] = point_filter( image, gauss_filter, i, j, 7 );
            }
        }
    }

    new_image.save( "result.png" );
}
