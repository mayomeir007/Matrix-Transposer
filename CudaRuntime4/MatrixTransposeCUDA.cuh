#pragma once

//main method. cretaes matrix with random values of size nx by ny, transposes it with the CPU and the GPU and prints performance information 
bool MatrixTransposeCUDA(int nx, int ny);

//initializes input array with random values between 0  and 255
void initialize(int* input, const int array_size);

//performs trnaspose on given matrix using the CPU 
void mat_transpose_cpu(int* mat, int* transpose, int nx, int ny);

//compares array a and array b. If one value is different, it prints that the arrays are different. Otherwise it prints that they are the same.
void compare_arrays(int* a, int* b, int size);

