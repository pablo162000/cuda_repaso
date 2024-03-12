#include <stdio.h>
#define image_channels 4

__device__ void process_pixel_min(unsigned char* src_image, unsigned char* dst_image, int width, int height, int x, int y) {


    unsigned char r_pixels[9];
    unsigned char g_pixels[9];
    unsigned char b_pixels[9];

    int cc = 0;

    for(int i=x-1;i<=x+1;i++) {
        for(int j=y-1;j<=y+1;j++) {
            int index = (j * width + i)*image_channels;

            if(i>=0 && i<width && j>=0 && j<height) {
                r_pixels[cc]=src_image[index];
                g_pixels[cc]=src_image[index + 1];
                b_pixels[cc]=src_image[index + 2];
                cc++;
            }
        }
    }
    unsigned int r=r_pixels[0];
    unsigned int g=g_pixels[0];
    unsigned int b=b_pixels[0];

    for(int i=1; i<cc;i++){
        r=min(r,r_pixels[i]);
        g=min(g,g_pixels[i]);
        b=min(b,b_pixels[i]);
    }

    int index = (y * width + x)*image_channels;
    dst_image[index+0] = r;
    dst_image[index+1] = g;
    dst_image[index+2] = b;
    dst_image[index+3] = 255;
}

__global__ void kerbel_min_image(unsigned char* src_image, unsigned char* dst_image, int width, int height) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel_min(src_image, dst_image, width, height, pix_x,pix_y);
}

extern "C" void kernel_min(unsigned char* src_image, unsigned char* dst_image, int width, int height) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kerbel_min_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height);
}

__device__ void process_pixel_max(unsigned char* src_image, unsigned char* dst_image, int width, int height, int x, int y) {


    unsigned char r_pixels[9];
    unsigned char g_pixels[9];
    unsigned char b_pixels[9];

    int cc = 0;

    for(int i=x-1;i<=x+1;i++) {
        for(int j=y-1;j<=y+1;j++) {
            int index = (j * width + i)*image_channels;

            if(i>=0 && i<width && j>=0 && j<height) {
                r_pixels[cc]=src_image[index];
                g_pixels[cc]=src_image[index + 1];
                b_pixels[cc]=src_image[index + 2];
                cc++;
            }
        }
    }
    unsigned int r=r_pixels[0];
    unsigned int g=g_pixels[0];
    unsigned int b=b_pixels[0];

    for(int i=1; i<cc;i++){
        r=max(r,r_pixels[i]);
        g=max(g,g_pixels[i]);
        b=max(b,b_pixels[i]);
    }

    int index = (y * width + x)*image_channels;
    dst_image[index+0] = r;
    dst_image[index+1] = g;
    dst_image[index+2] = b;
    dst_image[index+3] = 255;
}

__global__ void kerbel_max_image(unsigned char* src_image, unsigned char* dst_image, int width, int height) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel_max(src_image, dst_image, width, height, pix_x,pix_y);
}

extern "C" void kernel_max(unsigned char* src_image, unsigned char* dst_image, int width, int height) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kerbel_max_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height);
}

__device__ void process_pixel_median(unsigned char* src_image, unsigned char* dst_image, int width, int height, int x, int y) {


    unsigned char r_pixels[9];
    unsigned char g_pixels[9];
    unsigned char b_pixels[9];

    int cc = 0;

    for(int i=x-1;i<=x+1;i++) {
        for(int j=y-1;j<=y+1;j++) {
            int index = (j * width + i)*image_channels;

            if(i>=0 && i<width && j>=0 && j<height) {
                r_pixels[cc]=src_image[index];
                g_pixels[cc]=src_image[index + 1];
                b_pixels[cc]=src_image[index + 2];
                cc++;
            }
        }
    }

    int indiceMediana= cc/2;

    unsigned int r=r_pixels[indiceMediana];
    unsigned int g=g_pixels[indiceMediana];
    unsigned int b=b_pixels[indiceMediana];



    int index = (y * width + x)*image_channels;
    dst_image[index+0] = r;
    dst_image[index+1] = g;
    dst_image[index+2] = b;
    dst_image[index+3] = 255;
}

__global__ void kerbel_median_image(unsigned char* src_image, unsigned char* dst_image, int width, int height) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel_median(src_image, dst_image, width, height, pix_x,pix_y);
}

extern "C" void kernel_median(unsigned char* src_image, unsigned char* dst_image, int width, int height) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kerbel_median_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height);
}