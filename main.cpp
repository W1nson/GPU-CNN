// #include <hip/hip_runtime.h> 

#include <iostream> 
#include "matrix.h"

using namespace std;

// using 16 bit float 

int main() {
	

	int row = 3; 
	int col = 3; 
	
	float **data;
	data = new float *[row];
	for(int i = 0; i < row; i++) {
		data[i] = new float [col];
		for(int j = 0; j < col; j++) {
			data[i][j] = 1;
		}
	}
	Matrix mat1(data, row, col);

	Matrix mat2(data, row, col); 

	mat1.printGrid();
	mat2.printGrid(); 

	Matrix mat3 = mat1.matmul(mat2); 

	mat3.printGrid(); 


	return 0 ;
}