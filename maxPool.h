#ifndef maxPool_H
#define maxPool_H

#include <hip/hip_runtime.h> 
#include "mat.h"
// #include "matmul.h" 


void maxpool2D(Matrix input, int poolsize, Matrix out);



#endif 