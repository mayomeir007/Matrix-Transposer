#include <stdlib.h>
#include <stdio.h>

#include "MatrixTransposeCUDA.cuh"


int main(int argc, char** argv)
{
    bool valid = false;
    if(argc > 1)
    {
        valid = MatrixTransposeCUDA(atoi(argv[1]), atoi(argv[2]));
    }
    if (!valid)
    {
        printf("Invalid input\n");
    }
    return 0;
}
