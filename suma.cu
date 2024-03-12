#include <stdio.h>

#define  image_channels 4


__device__ void suma_pixel(unsigned char* src_image, unsigned char* dst_image, int width, int x, int y) {

    int index = (y * width + x)*image_channels;
    auto r = (src_image[index] + dst_image[index])/2;

    auto g = (src_image[index+1] + dst_image[index+1])/2;
    auto b = (src_image[index+2] + dst_image[index+2])/2;
    dst_image[index+0] = r;
    dst_image[index+1] = g;
    dst_image[index+2] = b;
    dst_image[index+3] = 255;
}


__global__ void kernel_suma_image(unsigned char* src_image, unsigned char * dst_image, int width){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int pix_y = index / width;
    int pix_x = index % width;

    suma_pixel(src_image,dst_image, width,pix_x,pix_y);

}


extern "C" void kernel_suma(unsigned char* src_image, unsigned char* dst_image, int width, int height) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kernel_suma_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width);
}