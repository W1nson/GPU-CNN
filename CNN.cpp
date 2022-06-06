#include <hip/hip_runtime.h>
#include <iostream>

#include "matmul.h"
#include "mat.h"
#include "conv2D.h"
#include "maxPool.h"
#include "dense.h" 

using namespace std;


void printMat(Matrix A) {
	cout << "Shape: " << A.height << 'x' << A.width << endl; 
	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            cout << A.elements[i*A.stride + j] << ' ';
    	}
		cout << endl;
	}
	cout << endl;
}



int main(){
	
	// CNN main layers demo 

	// 6 x 6 picture 
	// going through 2 conv2D and 1 maxpooling and 1 dense layer 
	int Cpu = 0; 
	int N = 6; 
	int M = 6; 
	int poolsize = 2; 

	int n = 4; 
	
    Matrix img(N, M, N, Cpu);
	Matrix filter1(3, 3, 0, Cpu), filter2(3, 3, 0, Cpu);
	Matrix conv(N, M, N, Cpu);
	Matrix pool(N/poolsize, M/poolsize, 0, Cpu); 
	Matrix W(n, pool.height, 0, Cpu);
	Matrix out(n, pool.width, 0, Cpu); 

	for( int i = 0; i < img.height; i++) {
    	for( int j = 0; j < img.width; j++) {
            img.elements[i*img.stride + j] = 1.0f; 
    	}
	}
	
	// straight edge detection
	filter1.elements[1] = 1;
	filter1.elements[4] = 1;
	filter1.elements[7] = 1;
	// {0, 1, 0, 0, 1, 0, 0, 1, 0}; 

	// diagnal edge detection 
	filter2.elements[0] = 1;
	filter2.elements[4] = 1; 
	filter2.elements[8] = 1; 
	// {1, 0, 0, 0, 1, 0, 0, 0, 1}; 

	cout << "filter1" << endl;
	printMat(filter1); 

	cout << "filter2" << endl;
	printMat(filter2); 

	cout << "img" << endl; 
	printMat(img); 

	cout << "First Conv2D" << endl; 
	conv2D(img,filter1,conv);

	printMat(conv);

	cout << "Second Conv2D" << endl; 
	conv2D(conv, filter2, conv);
	
	printMat(conv);

	cout << endl << "MaxPooling2D" << endl; 
	maxpool2D(conv, 2, pool);

	printMat(pool); 

	cout << "Dense Layer" << endl;
	dense(pool, "", W, out); 

	printMat(out);


	img.dealloc();
	filter1.dealloc();  
	filter2.dealloc(); 
	conv.dealloc(); 
	pool.dealloc();
	W.dealloc(); 
	out.dealloc();
	return 0; 

}







