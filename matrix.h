#ifndef MATRIX_H
#define MATRIX_H

// #include <hip/hip_runtime.h> 
// template <class float>
class Matrix { 
	
public: 
	int row, col; 
	int size[2]; 
	float **mat; 

	Matrix(); 
	Matrix(float **temp, int row, int col); 
	Matrix(int row, int col); 
	Matrix(float temp, int row, int col); 
	
	void random(); 

	Matrix add(Matrix temp); 
	Matrix subtract(Matrix temp); 

	Matrix multi(Matrix temp); 
	Matrix matmul(Matrix temp, float num = 0.0);

	void scale(float temp);

	float conv(Matrix temp); 
	void modi(std::string function);

	void transpose(); 
	void setSquare(); 
	
	int* getSize(); 
	
	void print();
	void printGrid();

};

#endif