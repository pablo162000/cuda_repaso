#include <stdio.h>

#define image_channels 4

__device__ void process_pixel(unsigned char* src_image, unsigned char* dst_image, int width, int x, int y) {
    int start =(y * width + x)*image_channels;
    int padding = x*image_channels;
    int end = ((y+1)*(width*image_channels)-1)-padding;
    dst_image[start]=src_image[end-3];
    dst_image[start+1]=src_image[end-2];
    dst_image[start+2]=src_image[end-1];
    dst_image[start+3]=src_image[end];
}

__global__ void kernel_espejo_image(unsigned char* src_image, unsigned char* dst_image, int width) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel(src_image, dst_image, width, pix_x,pix_y);
}

extern "C" void kernel_espejo(unsigned char* src_image, unsigned char* dst_image, int width, int height) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kernel_espejo_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width);
}