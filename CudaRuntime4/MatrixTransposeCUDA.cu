
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include "MatrixTransposeCUDA.cuh"

using namespace std::chrono;

__global__ void warm_up_kernel(int* mat, int* transpose, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        transpose[ix * ny + iy] = mat[iy * nx + ix];
    }
}

__global__ void transpose_unroll4_col(int* mat, int* transpose, int ny, int nx)
{
    int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int ti = iy * nx + ix;
    int to = ix * ny + iy;

    if (ix < nx && iy < ny)
    {
        transpose[ti] = mat[to];
    }

    if (ix + blockDim.x < nx && iy < ny)
    {
        transpose[ti + blockDim.x] = mat[to + blockDim.x * ny];
    }

    if (ix + 2 * blockDim.x < nx && iy < ny)
    {
        transpose[ti + 2 * blockDim.x] = mat[to + 2 * blockDim.x * ny];
    }

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        transpose[ti + 3 * blockDim.x] = mat[to + 3 * blockDim.x * ny];
    }
}

bool MatrixTransposeCUDA(int nx, int ny)
{
    if (nx < 1 || nx > 20000 || ny < 1 || ny > 20000)
    {
        return false;
    }

    int block_x = 128;
    int block_y = 8;

    int size = nx * ny;
    int byte_size = sizeof(int) * size;

    printf("Matrix transpose for %d X %d matrix \n", nx, ny);

    int* h_mat_array = (int*)malloc(byte_size);
    int* h_trans_array = (int*)malloc(byte_size);
    int* h_ref = (int*)malloc(byte_size);

    //initialize matrix with integers between 0 and 255
    initialize(h_mat_array, size);
    //matirx transpose in CPU
    //clock_t cpu_start, cpu_end;
    //cpu_start = clock();
    auto cpu_start = high_resolution_clock::now();
    mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);
    auto cpu_end = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(cpu_end - cpu_start);
    //cpu_end = clock();

    //printf("CPU execution time : %4.6f \n",(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

    int* d_mat_array, * d_trans_array;

    cudaMalloc((void**)&d_mat_array, byte_size);
    cudaMalloc((void**)&d_trans_array, byte_size);

    dim3 blocks(block_x, block_y);

    unsigned int transGridDimX = (ny + block_x - 1) / block_x;
    unsigned int transGridDimY = (nx + block_y - 1) / block_y;
    dim3 grid_col_unroll((transGridDimX + 3) / 4, transGridDimY);
    /////warm up /////////////////////////////////////
    //cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);

    //transpose_unroll4_col << < grid_col_unroll, blocks >> > (d_mat_array, d_trans_array, nx, ny);

    //cudaDeviceSynchronize();

    ////copy the transpose memroy back to cpu
    //cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost);
    //cudaMemset(d_trans_array, 0, byte_size);
    ///////////////////////////////////
    cudaEvent_t start, end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);

    transpose_unroll4_col << < grid_col_unroll, blocks >> > (d_mat_array, d_trans_array, nx, ny);

    cudaDeviceSynchronize();

    //copy the transpose memroy back to cpu
    cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time;
    cudaEventElapsedTime(&time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    //compare the CPU and GPU transpose matrix for validity
    compare_arrays(h_ref, h_trans_array, size);

    printf("CPU execution time : %f seconds\n", duration.count()/1000000.f);

    printf("GPU execution time : %f seconds\n", time/1000.f);

    float gpu_speed_factor = (duration.count() / time) / 1000.f;
    printf("GPU executed the matrix transpose %f times faster. \n", gpu_speed_factor);
    if (gpu_speed_factor > 2)
    {
        printf("Doing more operations on to the GPU would be a good idea.\n");
    }
    else
    {
        printf("Sorry the capability of the GPU didn't impress you. \nIf you havent ran this program in a while, consider running it again a few more times to warm up the GPU.\n");
    }
    cudaDeviceReset();
    return true;
}

void initialize(int* input, const int array_size)
{
    srand(time(0));
    for (int i = 0; i < array_size; i++)
    {
        input[i] = (int)(rand() & 0xFF);
    }
}

void mat_transpose_cpu(int* mat, int* transpose, int nx, int ny)
{
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            transpose[ix * ny + iy] = mat[iy * nx + ix];
        }
    }
}

void compare_arrays(int* a, int* b, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            printf("Array transpose using CPU and array transpose using GPU are different! \n");
            printf("at index %d - CPU value %d | value GPU %d \n", i, a[i], b[i]);
            return;
        }
    }
    printf("Array transposed using CPU and array transposed using GPU are the same :) \n");
}