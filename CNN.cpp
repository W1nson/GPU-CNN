#include <hip/hip_runtime.h>
#include <iostream>

#include "matmul.h"
#include "mat.h"
#include "conv2D.h"

using namespace std;

int main(){

	int Cpu = 0; 
	int N = 4; 
	// int M = 5; 

    Matrix A(N, N, N, Cpu), B(3, 3, 0, Cpu), C(N, N, N, Cpu);

	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            A.elements[i*A.stride + j] = 1.0f;
           
    	}
	}
	for(int i = 0; i < B.height; i++) {
    	for(int j = 0; j < B.width; j++) {
	 		B.elements[i*B.stride + j] = 1.0f;
    	}
	}

	cout << 'A' << endl;
	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            cout << A.elements[i*A.stride + j] << ' ';
    	}
		cout << endl;
	}
	cout << 'B' << endl;
	for( int i = 0; i < B.height; i++) {
    	for( int j = 0; j < B.width; j++) {
            cout << B.elements[i*B.stride + j] << ' ';
    	}
		cout << endl;
	}

	// serial_convolution(A, B, C);
	// cout << 'C' << endl;
	// for( int i = 0; i < C.height; i++) {
    // 	for( int j = 0; j < C.width; j++) {
    //         cout << C.elements[i*C.stride + j] << ' ';
    // 	}
	// 	cout << endl;
	// }	
	conv2D(A,B,C);
	
	cout << 'C' << endl;
	for( int i = 0; i < C.height; i++) {
    	for( int j = 0; j < C.width; j++) {
            cout << C.elements[i*C.stride + j] << ' ';
    	}
		cout << endl;
	}

	A.dealloc(); 
	B.dealloc(); 
	C.dealloc();
	return 0; 
// 	// Set up matrices
//     int Cpu = 0;
//     int N = 3;
//     int M = 3;

//     Matrix A(N, M, N, Cpu), B(M, N, M, Cpu), C(N, N, N, Cpu);

// 	//set values for A and B 
//     for( int i = 0; i < A.height; i++) {
//     	for( int j = 0; j < A.width; j++) {
//             A.elements[i*A.stride + j] = 1.0f;
//             B.elements[i*B.stride + j] = 2.0f;
//     	}
// 	}

// 	cout << 'A' << endl;
// 	for( int i = 0; i < A.height; i++) {
//     	for( int j = 0; j < A.width; j++) {
//             cout << A.elements[i*A.stride + j] << ' ';
//     	}
// 		cout << endl;
// 	}
// 	cout << 'B' << endl;
// 	for( int i = 0; i < B.height; i++) {
//     	for( int j = 0; j < B.width; j++) {
//             cout << B.elements[i*B.stride + j] << ' ';
//     	}
// 		cout << endl;
// 	}
// //SharedMemHIP
// 	MatMul(A,B,C);
	
// 	cout << 'C' << endl;
// 	for( int i = 0; i < C.height; i++) {
//     	for( int j = 0; j < C.width; j++) {
//             cout << C.elements[i*C.stride + j] << ' ';
//     	}
// 		cout << endl;
// 	}

// //Deallocate Memory
//     A.dealloc();
//     B.dealloc();
//     C.dealloc();
}







