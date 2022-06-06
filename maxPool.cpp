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

	// check if share id is bigger than the pool block 
	// load the data on to block
	if(sid  < ssize) {
		sdata[sid] = input.elements[row * input.stride + col];  
	}
	__syncthreads(); // make sure all the load finishes 


	// using reduce idea to reduce the data in the block
	for(int s = ssize/2; s>0; s >>=1) {
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

	// this is the naive way of find max of each window. 
	// if(sid == 0) {
	// 	float max = sdata[0]; 
	// 	for(int k = 1; k < ssize; k++) {
	// 		if(sdata[k] > max) 
	// 			max = sdata[k];
	// 	}
	// 	out.elements[blockIdx.y * out.stride + blockIdx.x] = max;
	// }	
}

// wrapper funciton for the device function
void maxpool2D(Matrix input, int poolsize, Matrix out)  {
	int Gpu = 1, toDev = 1, fromDev = 2; 

	Matrix d_A(input.width, input.height, 0, Gpu); 
	Matrix d_B(out.width, out.height, 0, Gpu); 

	d_A.load(input, toDev); 
	
	dim3 dimBlock(poolsize, poolsize); 
	dim3 dimGrid(input.width/dimBlock.x, input.height/dimBlock.y); 

	cout << "dimBlock: " << dimBlock.y << 'x' << dimBlock.x << endl; 
	cout << "dimGrid: " << dimGrid.y << 'x' << dimGrid.x << endl; 
	hipEvent_t start, stop; 
    float elapsed_secs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0); 

	max_reduce<<<dimGrid, dimBlock, poolsize*poolsize>>>(d_A, poolsize, d_B); 

	hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_secs, start, stop); 
    cout<<"GPU MaxPooling Time = "<< elapsed_secs << "ms" << endl;
   

	out.load(d_B, fromDev); 

	d_A.dealloc(Gpu); 
	d_B.dealloc(Gpu); 

}


// serial_maxpooling is demonstrating the idea of maxpooling 
void serial_maxpool2D(Matrix input, int poolsize, Matrix out) {
	int Cpu = 0; 
	
	
	Matrix A(input.width, input.height, 0, Cpu);
	A.load(input); 
	Matrix B(out.width, out.height, 0, Cpu);
	
	
	float temp[10] = {0};

	clock_t begin = clock();
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

	clock_t end = clock();
	double fullcpu = double(end - begin) / (CLOCKS_PER_SEC*12);
	std::cout<< " CPU Time = " << fullcpu << "s" << std::endl; 

	out.load(A);

	
	A.dealloc();
	B.dealloc();
}


// int main () {

// 	int Cpu = 0; 
// 	int N = 2048; 
// 	int poolsize = 2;
// 	Matrix A(N, N, N, Cpu), C(N/poolsize, N/poolsize, 0, Cpu), D(N/poolsize, N/poolsize, 0, Cpu);
// 	for( int i = 0; i < A.height; i++) {
//     	for( int j = 0; j < A.width; j++) {
//             A.elements[i*A.stride + j] = i+j;
//     	}
// 	}

// 	cout << 'A' << endl;
// 	for( int i = 0; i < A.height; i++) {
//     	for( int j = 0; j < A.width; j++) {
//             cout << A.elements[i*A.stride + j] << ' ';
//     	}
// 		cout << endl;
// 	}
// 	cout << endl; 


// 	// serial_maxpool2D(A, poolsize, C); 
// 	// cout << "Serial C" << endl; 
// 	// for( int i = 0; i < C.height; i++) {
//     // 	for( int j = 0; j < C.width; j++) {
//     //         cout << C.elements[i*C.stride + j] << ' ';
//     // 	}
// 	// 	cout << endl;
// 	// }
// 	// cout << endl; 


// 	maxpool2D(A, poolsize, D); 
// 	cout << "Gpu D" << endl; 
// 	for( int i = 0; i < D.height; i++) {
//     	for( int j = 0; j < D.width; j++) {
//             cout << D.elements[i*D.stride + j] << ' ';
//     	}
// 		cout << endl;
// 	}
// 	cout << endl; 
// }