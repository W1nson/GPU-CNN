#ifndef DENSE_H
#define DENSE_H

#include <hip/hip_runtime.h> 
#include "mat.h"
// #include "matmul.h" 


void dense_serial(Matrix A, std::string activation, Matrix W, Matrix out);
void dense(Matrix A, std::string activation, Matrix B, Matrix C);


#endif 