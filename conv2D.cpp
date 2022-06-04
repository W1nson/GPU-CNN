#include <iostream> 
#include <hip/hip_runtime.h> 
// #include "matmul.h" 
#include "mat.h" 

#define BLOCK_SIZE 32

using namespace std; 

__global__ void convolution(const Matrix input, const Matrix filter, Matrix output) {
	
	int row = blockIdx.y * blockDim.y + threadIdx.y+1; // 1
	int col = blockIdx.x * blockDim.x + threadIdx.x+1; // 1
	if(row > input.height-1 || col > input.width-1) 
		return; 

	float sum = 0.0f; 
	
	for(int i = -1; i < filter.height-1; i++) {
		for(int j = -1; j < filter.width-1; j++) {
			sum += input.elements[(i+row) * input.stride + (j+col)] * filter.elements[(i+1) * filter.stride + (j+1)];
		}
	}

	__syncthreads();
	output.elements[(row - 1) * output.stride + (col - 1)] = sum;

}

__global__ void convolution_shared(const Matrix input, const Matrix filter, Matrix output) {
	
	int tid = threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y+1; // 1
	int col = blockIdx.x * blockDim.x + threadIdx.x+1; // 1
	
	extern __shared__ float sdata[];

	// if the center if outside of the bounds don't need to do anything 
	if(row > input.height-1 || col > input.width-1) 
		return; 


	if(tid < filter.height * filter.width)
		sdata[tid] = filter.elements[tid]; 
	__syncthreads(); 

	float sum = 0.0f; 
	for(int i = -1; i < filter.height-1; i++) {
		for(int j = -1; j < filter.width-1; j++) {
			sum += input.elements[(i+row) * input.stride + (j+col)] * sdata[(i+1) * filter.stride + (j+1)];
		}
	}
	// for(int i = -1; i < filter.height-1; i++) {
	// 	for(int j = -1; j < filter.width-1; j++) {
	// 		sum += input.elements[(i+row) * input.stride + (j+col)] * filter.elements[(i+1) * filter.stride + (j+1)];
	// 	}
	// }

	__syncthreads();
	output.elements[(row - 1) * output.stride + (col - 1)] = sum;

}
void conv2D(const Matrix input, const Matrix filter, Matrix out) {

	// without padding edges are 0 
	int Cpu = 0, Gpu = 1, toDev = 1, fromDev = 2; 
	int kernalsize = filter.width * filter.height; 

	Matrix A(input.width+2,input.height+2, 0, Cpu);
	for(int i = 0; i < A.height; i++) {
		for(int j = 0; j < A.width; j++) {
			A.elements[i*A.stride + j] = 0;
		}
	}	
	for(int i = 1; i < A.height-1; i++) {
		for(int j = 1; j < A.width-1; j++) {
			A.elements[i*A.stride + j] = input.elements[(i-1)*input.stride + (j-1)];
		}
	}
	cout << "A\'" << endl;
	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            cout << A.elements[i*A.stride + j] << ' ';
    	}
		cout << endl;
	}	
	Matrix d_A(A.width,A.height, 0, Gpu);
	Matrix d_B(filter.width, filter.height, 0, Gpu);
	Matrix d_C(out.width, out.height, 0, Gpu);

	d_A.load(A, toDev);
	d_B.load(filter, toDev);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
	dim3 dimGrid(input.width / dimBlock.x +1, input.height / dimBlock.y +1);


	cout << "dimBlock: " << dimBlock.y << 'x' << dimBlock.x << endl; 
	cout << "dimGrid: " << dimGrid.y << 'x' << dimGrid.x << endl; 
	

	convolution<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 

	out.load(d_C, fromDev);


	A.dealloc();
	d_A.dealloc(Gpu); 
	d_B.dealloc(Gpu); 
	d_C.dealloc(Gpu); 

}

void serial_convolution(const Matrix input, const Matrix filter, Matrix output) {
	int Cpu = 0;	
	Matrix A(input.width+2,input.height+2, 0, Cpu);
	Matrix B(input.width, input.height, 0, Cpu); 

	for(int i = 1; i < A.height-1; i++) {
		for(int j = 1; j < A.width-1; j++) {
			A.elements[i*A.stride + j] = input.elements[(i-1)*input.stride + (j-1)];
		}
	}
	// for( int i = 0; i < A.height; i++) {
    // 	for( int j = 0; j < A.width; j++) {
    //         cout << A.elements[i*A.stride + j] << ' ';
    // 	}
	// 	cout << endl;
	// }	

	for( int i = 1; i < A.height-1; i++) {
    	for( int j = 1; j < A.width-1; j++) {
			float sum = 0;
			for(int row = -1; row < filter.height-1; row++) {
				for(int col = -1; col < filter.width-1; col++) {
					sum += A.elements[(i+row)*A.stride + (j+col)] * filter.elements[(row+1)*filter.stride + (col+1)];
				}
			}
			B.elements[(i - 1) * B.stride + (j - 1)] = sum;
		}
	}
	output.load(B); 
	B.dealloc();
	A.dealloc();
}




int main () { 

	int Cpu = 0; 
	int N = 4; 
	// int M = 4; 

    Matrix A(N, N, 0, Cpu), B(3, 3, 0, Cpu), C(N, N, 0, Cpu);

	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            A.elements[i*A.stride + j] = 1.0f;
            
    	}
	}
	for( int i = 0; i < B.height; i++) {
    	for( int j = 0; j < B.width; j++) {
			B.elements[i*B.stride + j] = 2.0f;
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

	serial_convolution(A, B, C);
	
	cout << "Serial C" << endl;
	for( int i = 0; i < C.height; i++) {
    	for( int j = 0; j < C.width; j++) {
            cout << C.elements[i*C.stride + j] << ' ';
    	}
		cout << endl;
	}

	conv2D(A,B,C);
	
	cout << "GPU C" << endl;
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
}


