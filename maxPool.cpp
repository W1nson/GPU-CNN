#include <iostream> 

#include <hip/hip_runtime.h>

#include "mat.h" 
#include "maxPool.h"

using namespace std; 

#define BLOCK_SIZE 32

// max_reduce for pool size 
__global__ void max_reduce(Matrix input, const int poolsize, Matrix out) {

	// shared memory to store the pool 
	extern __shared__ float sdata[];  // 2x2 pool, size 4 


	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x; 


	int ssize = poolsize * poolsize;
	int sid = threadIdx.y * blockDim.y + threadIdx.x; 
	if(sid  < ssize) {
		sdata[sid] = input.elements[row * input.stride + col];  
	}
	__syncthreads();

	int s = ssize/2;
	if(poolsize %2 != 0) 
		s = (ssize)/2; 

	for(; s>0; s >>=1) {
        if(sid < s) {
            if(sdata[sid] > sdata[sid + s]) 
                sdata[sid] =  sdata[sid];    
            else
                sdata[sid] = sdata[sid+s]; 
        }
        __syncthreads(); //make sure all comparison is finish 
    }
	
	if(sid == 0) {	
		int max = sdata[0];
		if(max < sdata[ssize-1])
			max = sdata[ssize-1];
		out.elements[blockIdx.y * out.stride + blockIdx.x] = max;
	}

	// if(sid == 0) {
	// 	float max = sdata[0]; 
	// 	for(int k = 1; k < ssize; k++) {
	// 		if(sdata[k] > max) 
	// 			max = sdata[k];
	// 	}
	// 	out.elements[blockIdx.y * out.stride + blockIdx.x] = max;
	// }

	
	
}


void maxpool2D(Matrix input, int poolsize, Matrix out)  {
	int Gpu = 1, toDev = 1, fromDev = 2; 

	Matrix d_A(input.width, input.height, 0, Gpu); 
	Matrix d_B(out.width, out.height, 0, Gpu); 

	d_A.load(input, toDev); 
	
	dim3 dimBlock(poolsize, poolsize); 
	dim3 dimGrid(input.width/dimBlock.x, input.height/dimBlock.y); 

	cout << "dimBlock: " << dimBlock.y << 'x' << dimBlock.x << endl; 
	cout << "dimGrid: " << dimGrid.y << 'x' << dimGrid.x << endl; 

	max_reduce<<<dimGrid, dimBlock, poolsize*poolsize>>>(d_A, poolsize, d_B); 


	out.load(d_B, fromDev); 

	d_A.dealloc(Gpu); 
	d_B.dealloc(Gpu); 


}



void serial_maxpool2D(Matrix input, int poolsize, Matrix out) {
	int Cpu = 0; 
	
	
	Matrix A(input.width, input.height, 0, Cpu);
	A.load(input); 
	Matrix B(out.width, out.height, 0, Cpu);
	
	
	float temp[10] = {0};
	for(int i = 0; i < A.height; i+= poolsize) {
		for(int j = 0; j < A.width; j+=poolsize) {
			if (poolsize == 2)
			{
				temp[0] = A.elements[i* A.stride + j];
				temp[1] = A.elements[i* A.stride + j + 1 ];
				temp[2] = A.elements[(i+1)* A.stride + j];
				temp[3] = A.elements[(i+1)* A.stride + j+1];
			}
			else if (poolsize == 3)
			{
				temp[0] = A.elements[i* A.stride + j];
				temp[1] = A.elements[i* A.stride + j + 1];
				temp[2] = A.elements[i* A.stride + j + 2];
				temp[3] = A.elements[(i+1)* A.stride + j];
				temp[4] = A.elements[(i+1)* A.stride + j+1];
				temp[5] = A.elements[(i+1)* A.stride + j+2];
				temp[6] = A.elements[(i+2)* A.stride + j];
				temp[7] = A.elements[(i+2)* A.stride + j+1];
				temp[8] = A.elements[(i+2)* A.stride + j+2];
			}

			float max = temp[0]; 
			for(int k = 1; k < 10; k++) {
				if(temp[k] > max) 
					max = temp[k];
			}
			A.elements[(i/poolsize)*A.stride + (j/poolsize)] = max; 
		}
	}
	out.load(A);

	
	A.dealloc();
	B.dealloc();
}



int main () {
	int Cpu = 0; 
	int N = 12; 
	int poolsize = 3;
	Matrix A(N, N, N, Cpu), C(N/poolsize, N/poolsize, 0, Cpu), D(N/poolsize, N/poolsize, 0, Cpu);;
	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            A.elements[i*A.stride + j] = i+j;
    	}
	}

	A.elements[13] = 10;

	cout << 'A' << endl;
	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            cout << A.elements[i*A.stride + j] << ' ';
    	}
		cout << endl;
	}
	cout << endl; 


	serial_maxpool2D(A, poolsize, C); 
	cout << "Serial C" << endl; 
	for( int i = 0; i < C.height; i++) {
    	for( int j = 0; j < C.width; j++) {
            cout << C.elements[i*C.stride + j] << ' ';
    	}
		cout << endl;
	}
	cout << endl; 


	maxpool2D(A, poolsize, D); 
	cout << "Gpu D" << endl; 
	for( int i = 0; i < D.height; i++) {
    	for( int j = 0; j < D.width; j++) {
            cout << D.elements[i*D.stride + j] << ' ';
    	}
		cout << endl;
	}
	cout << endl; 
}