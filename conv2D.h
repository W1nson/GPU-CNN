#ifndef CONV2D_H
#define CONV2D_H

#include <hip/hip_runtime.h> 
#include "mat.h"
// #include "matmul.h" 


void serial_convolution(Matrix input, Matrix filter, Matrix output);

void conv2D(const Matrix input, const Matrix filter, Matrix out);



#endif 