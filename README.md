This is a CUDA console application that accepts matrix dimensions nx and ny, creates a nx by ny matrix with random values and performs matrix transpose operation on it using the CPU and then GPU. It then compares the time each method took. Accordng to the table below, matrices with nx and ny higher than 512 a better transposed with the GPU. The specific GPU sceme used to perform the transpose is column major with unrolling factor of 4. 
This methd is possible only when each matrix element value is not a function of other matix element values.

![image](https://github.com/user-attachments/assets/9261309c-000e-408f-bb17-a1ed0a90588e)




