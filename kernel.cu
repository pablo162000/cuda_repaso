#include <stdio.h>

#define image_channels 4

__device__ void process_pixel(unsigned char* src_image, unsigned char* dst_image, int width, int height, int x, int y, int blur_step) {

    /**
     * (x-1,y-1)  (x,y+1)  (x+1,y+1)
     * (x-1,y)    (x,y)    (x+1,y)
     * (x-1,y+1)  (x,y+1)  (x+1,y-1)
     */

    int r = 0;
    int g = 0;
    int b = 0;
    int cc = 0;

    for(int i=x-blur_step;i<=x+blur_step;i++) {
        for(int j=y-blur_step;j<=y+blur_step;j++) {
            int index = (j * width + i)*image_channels;

            //if(index>=0 && index<width*height*image_channels) {
            if(i>=0 && i<width && j>=0 && j<height) {
                r = r + src_image[index];
                g = g + src_image[index + 1];
                b = b + src_image[index + 2];
                cc++;
            }
        }
    }

    int index = (y * width + x)*image_channels;

    dst_image[index+0] = r/cc;
    dst_image[index+1] = g/cc;
    dst_image[index+2] = b/cc;
    dst_image[index+3] = 255;
}

__global__ void kerbel_blur_image(unsigned char* src_image, unsigned char* dst_image, int width, int height, int blur_step) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel(src_image, dst_image, width, height, pix_x,pix_y, blur_step);
}

extern "C" void kernel_blur(unsigned char* src_image, unsigned char* dst_image, int width, int height, int blur_step) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kerbel_blur_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height, blur_step);
}