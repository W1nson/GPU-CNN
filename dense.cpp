#include <iostream> 
#include <hip/hip_runtime.h> 
#include <cstdlib> 
#include "mat.h" 


using namespace std; 

#define SEED 0

// BLOCK_SIZE has to be smaller than number of weights 
#define BLOCK_SIZE 1


// setting the random weight to the weight matrix (can be vector) 
void load_rand(Matrix w) {
	for(int i = 0; i < w.height; i++) {
		for(int j = 0; j < w.width; j++) {
			w.elements[i*w.stride + j] = (float) rand()/RAND_MAX;
			// w.elements[i*w.stride + j] = 1;
	
		}
	}
}

// GPU Naive matmul 
__global__ void matmul(const Matrix A, const Matrix B, Matrix C)
{
	
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x; 
	if(row > A.height || col > A.width)
		return; 
	float sum = 0.0f;
	for (int e = 0; e<A.width; e++)
		sum += A.elements[row*A.width + e]*B.elements[e*B.width + col];

	C.elements[row*C.width + col] = sum;
}


// __global__ void sigmoid(const Matrix A, Matrix B) {
// 	int row = blockIdx.y*blockDim.y + threadIdx.y;
// 	int col = blockIdx.x*blockDim.x + threadIdx.x;

// 	if(row > A.height-1 || col > A.width-1)
// 		return;  
// 	B.elements[row*B.stride + col] = (float)1 / (1 + exp(A.elements[row * A.stride + col]));	
// }


// __global__ void relu(const Matrix A, Matrix B) {
// 	int row = blockIdx.y*blockDim.y + threadIdx.y;
// 	int col = blockIdx.x*blockDim.x + threadIdx.x; 
// 	if (A.elements[row*A.stride + col] < 0)	
// 		B.elements[row*B.stride + col] = 0;
// }

// __global__ void leakyrelu(const Matrix A, Matrix B) {
// 	int row = blockIdx.y*blockDim.y + threadIdx.y;
// 	int col = blockIdx.x*blockDim.x + threadIdx.x; 
// 	if (A.elements[row*A.stride + col] < 0)  
// 		B.elements[row*B.stride + col] = float(0.01) * A.elements[row*A.stride + col];	
// }




void dense(Matrix A, string activation, Matrix B, Matrix C)
{
    int Gpu=1, toDev = 1, fromDev = 2; 
	//Load A and B to device memory
	load_rand(B);
    Matrix d_A(A.width, A.height,0, Gpu);
    d_A.load(A, toDev); 
    Matrix d_B(B.width, B.height,0, Gpu);
    d_B.load(B, toDev); 

	// Allocate C in device memory
    Matrix d_C(C.width, C.height,0, Gpu);
    

    // Invoke kernel 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    // Use hipEvent type for timing

    hipEvent_t start, stop; 
    float elapsed_secs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0); 

    matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// update the sum with activation function 
	// if (activation.compare("sigmoid") == 0) {
	// 	sigmoid<<<dimGrid, dimBlock>>>(d_C, d_C);
	// }
	// else if (activation.compare("relu") == 0) {
	// 	relu<<<dimGrid, dimBlock>>>(d_C, d_C);
	// }
	// else if (activation.compare("leakyrelu") == 0) {
	// 	leakyrelu<<<dimGrid, dimBlock>>>(d_C, d_C);
	// }

    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_secs, start, stop); 
    std::cout<<"Naive GPU MatMul Time = "<< elapsed_secs << "ms" << std::endl;
    // Read C from device memory 
    C.load(d_C, fromDev); 
    // Free device memory 
    d_A.dealloc(Gpu);
    d_B.dealloc(Gpu);
    d_C.dealloc(Gpu); 
}


void dense_serial(Matrix A, string activation, Matrix W, Matrix out) {
	
	srand(SEED);

	int Cpu = 0; 
	// randomize the w matrix to have the 

	Matrix w(W.width, W.height, 0, Cpu); 

	Matrix B(out.width, out.height, 0, Cpu); 
	for(int i = 0; i < w.height; i++) {
		for(int j = 0; j < w.width; j++) {
			// w.elements[i * w.stride + j]  = 1; 
			w.elements[i * w.stride + j] = (float) rand()/RAND_MAX; 
		}

	}	
	clock_t begin = clock();		
	// matmul
	for(int i = 0; i < B.height; i++) {
		for(int j = 0; j < B.width; j++) {
			float sum = 0.0f;
			for(int k = 0; k < A.width; k++) 
				sum += A.elements[i*A.stride + k] * w.elements[k*w.stride+j];
			
			if (activation.compare("sigmoid") == 0) {
				sum = 1 / (1 + exp(sum));
			}
			else if (activation.compare("relu") == 0) {
				if (sum < 0)
					sum = 0;
			}

			else if (activation.compare("leakyrelu") == 0) {
				if (sum < 0)
					sum = float(0.01) * sum;

			}
			B.elements[ i*B.stride + j] = sum;
		}
	}
	clock_t end = clock();
	double fullcpu = double(end - begin) / (CLOCKS_PER_SEC*12);
	std::cout<< " CPU Time = " << fullcpu << "s" << std::endl; 
	

	// only works for output layer which should have one column with size of classes
	if (activation.compare("softmax") == 0) {
		if(B.width == 1) {
			float temp = 0; 
			for (int i = 0; i < B.height; i++) {
				temp += exp(B.elements[i]);
			}			
		}
	}		 


	
	out.load(B); 
	W.load(w);

}





// int main () {

// 	int Cpu = 0; 
// 	int N = 1024; 
// 	int M = 1024;
// 	int num = 64; 

// 	srand(SEED);


// 	Matrix A(N, M, 0, Cpu);
// 	Matrix W(num, A.width, 0,Cpu);
// 	Matrix B(num, A.height, 0, Cpu);
	

// 	for( int i = 0; i < A.height; i++) {
//     	for( int j = 0; j < A.width; j++) {
//             A.elements[i*A.stride + j] = i+j;
//     	}
// 	}

// 	// cout << "A shape: " << A.height << 'x' << A.width << endl;
// 	// for( int i = 0; i < A.height; i++) {
//     // 	for( int j = 0; j < A.width; j++) {
//     //         cout << A.elements[i*A.stride + j] << ' ';
//     // 	}
// 	// 	cout << endl;
// 	// }

// 	// dense_serial( A, "leakyrelu",  W, B);
// 	dense(A, "sigmoid", W, B); 
// 	cout << "B shape: " << B.height << 'x' << B.width << endl;
// 	// for( int i = 0; i < B.height; i++) {
//     	for( int j = 0; j < B.width; j++) {
//             cout << B.elements[0*B.stride + j] << ' ';
//     	}
// 		cout << endl;
// 	// }

// 	// cout << "W shape: " << W.height << 'x' << W.width <<endl;
// 	// for( int i = 0; i < W.height; i++) {
//     // 	for( int j = 0; j < W.width; j++) {
//     //         cout << W.elements[i*W.stride + j] << ' ';
//     // 	}
// 	// 	cout << endl;
// 	// }

	

// }