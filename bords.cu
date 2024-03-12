#include <stdio.h>

#define image_channels 4

__device__ void process_pixel(unsigned char* src_image, unsigned char* dst_image, int width, int height, int x, int y) {


    int r = 0;
    int g = 0;
    int b = 0;
    int cc = 0;

    int matriz[] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };

    for(int i=-1;i<=1;i++) {
        for(int j=-1;j<=1;j++) {
            int index = (y * width + x)*image_channels + (i * 4) + (j * width * 4);

            if(index >= 0 && index <= width * height * image_channels) {
                int matrixIndex = (i + 1) * 3 + (j + 1);
                int weight = matriz[matrixIndex];

                r += src_image[index] * weight;
                g += src_image[index + 1] * weight;
                b += src_image[index + 2] * weight;
            }
        }
    }

    int index = (y * width + x)*image_channels;
    r = max(0,min(r,255));
    g = max(0,min(g,255));
    b = max(0,min(b,255));

    dst_image[index+0] = r;
    dst_image[index+1] = g;
    dst_image[index+2] = b;
    dst_image[index+3] = 255;
}

__global__ void kerbel_bordes_image(unsigned char* src_image, unsigned char* dst_image, int width, int height) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel(src_image, dst_image, width, height, pix_x,pix_y);
}

extern "C" void kernel_bordes(unsigned char* src_image, unsigned char* dst_image, int width, int height) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kerbel_bordes_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height);
}

__device__ void process_pixel2(unsigned char* src_image, unsigned char* dst_image, int width, int height, int x, int y) {

    int r_x = 0;
    int g_x = 0;
    int b_x = 0;

    int r_y = 0;
    int g_y = 0;
    int b_y = 0;

    int sobel_x[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int sobel_y[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int index = (y * width + x)*image_channels + (i * 4) + (j * width * 4);

            if (index >= 0 && index <= width * height * 4) {
                int matrixIndex = (i + 1) * 3 + (j + 1);

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

__global__ void kerbel_bordes_image2(unsigned char* src_image, unsigned char* dst_image, int width, int height) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel2(src_image, dst_image, width, height, pix_x,pix_y);
}

extern "C" void kernel_bordes2(unsigned char* src_image, unsigned char* dst_image, int width, int height) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kerbel_bordes_image2<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height);
}