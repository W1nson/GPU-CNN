#include <hip/hip_runtime.h> 

#include <iostream> 
#include "matrix.h"
#include "matmul.h" 

using namespace std;

// using 16 bit float 

int main() {
	

	size_t row = 3; 
	size_t col = 3; 
	size_t stride = 3;
	
	float *data;
	data = new float [row*col];
	for(size_t i = 0; i < row; i++) {
		for(size_t j = 0; j < col; j++) {
			data[i * stride + j] = 0.1f;
		}
	}
	Matrix mat1(data, row, col);

	Matrix mat2(data, row, col); 

	mat1.printGrid();
	mat2.printGrid(); 

	Matrix mat3 = mat1.matmul(mat2); 

	mat3.printGrid(); 

	mat3 = mat1.subtract(mat2); 
	
	mat3.printGrid(); 

	mat3 = mat1.add(mat2); 
	
	mat3.printGrid();

	mat3 = mat1.multi(mat2); 
	
	mat3.printGrid();  

	return 0 ;
}