#include <stdio.h>

#define image_channels 4

__global__ void kernel_rotar_image(unsigned char* src_image, unsigned char* dst_image, int width, int height, int blur_step) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int x = index % width;
    int y = index / width;

    int id = (y * width + x) * 4;
    int rotatedIndex = ((width - x - 1) * height + y) * 4;

    dst_image[rotatedIndex] = src_image[id];
    dst_image[rotatedIndex + 1] = src_image[id + 1];
    dst_image[rotatedIndex + 2] = src_image[id + 2];
    dst_image[rotatedIndex + 3] = src_image[id + 3];
}


extern "C" void kernel_rotar(unsigned char* src_image, unsigned char* dst_image, int width, int height, int blur_step) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kernel_rotar_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height, blur_step);
}