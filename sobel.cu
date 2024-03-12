#include <stdio.h>
#include <iostream>
#include <fmt/core.h>

#define image_channels 4

__device__ void process_pixel_sobel(unsigned char* src_image, unsigned char* dst_image, int width, int height, int x, int y) {

    int r_x = 0;
    int g_x = 0;
    int b_x = 0;

    int r_y = 0;
    int g_y = 0;
    int b_y = 0;

    int sobel_x[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int sobel_y[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

    for (int i = x-1; i <= x+1; i++) {
        for (int j = y-1; j <=y+ 1; j++) {
            int index = (j * width + i)*image_channels;

            if (index >= 0 && index <= width * height * 4) {
                int matrixIndex = (i-x + 1) * 3 + (j-y + 1);

                int weight_x = sobel_x[matrixIndex];
                int weight_y = sobel_y[matrixIndex];

                r_x += src_image[index] * weight_x;
                g_x += src_image[index + 1] * weight_x;
                b_x += src_image[index + 2] * weight_x;

                r_y += src_image[index] * weight_y;
                g_y += src_image[index + 1] * weight_y;
                b_y += src_image[index + 2] * weight_y;
            }
        }
    }

    int r = max(0,min((abs(r_x) + abs(r_y))/2,255));
    int g = max(0,min((abs(g_x) + abs(g_y))/2,255));
    int b = max(0,min((abs(b_x) + abs(b_y))/2,255));

    int index = (y * width + x)*image_channels;
    dst_image[index+0] = r;
    dst_image[index+1] = g;
    dst_image[index+2] = b;
    dst_image[index+3] = 255;
}

__global__ void kerbel_blur_image(unsigned char* src_image, unsigned char* dst_image, int width, int height) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel_sobel(src_image, dst_image, width, height, pix_x,pix_y);
}

extern "C" void kernel_sobel(unsigned char* src_image, unsigned char* dst_image, int width, int height) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kerbel_blur_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height);
}