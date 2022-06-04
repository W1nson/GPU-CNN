#ifndef MATMUL_H
#define MATMUL_H

#include <hip/hip_runtime.h> 
#include "mat.h"

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix); 
void NaiveMatMul(const Matrix A, const Matrix B, Matrix C);
void MatMul(const Matrix A, const Matrix B, Matrix C);
#endif 