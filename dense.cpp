#include <iostream> 
#include <hip/hip_runtime.h> 
#include <cstdlib> 
#include "mat.h" 
#include "matmul.h" 
#include "submat.h" 

using namespace std; 

#define SEED 1
#define BLOCK_SIZE 16


__global__ void matmul(Matrix A, Matrix B, Matrix C)
{
	//Static shared memory for Asub and Bsub
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE]; //Great name for an array


	// Block row and column;
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	//Thread block computes one sub matrix Csub of C
	subMatrix Csub(C, BLOCK_SIZE,  blockRow, blockCol);

	// Each thread computes one element of Csub
	// By accumulating results into Cvalue
	float Cvalue = 0.0f; 

	//Thread row and column index within the submatrix
	int row = threadIdx.y;
	int col = threadIdx.x; 

	// Loop over submatrices of A and B that are required for Csub
	//Multiply each pair of sub-matrices together
	//and summ the results
	for (int m = 0; m < (A.width/BLOCK_SIZE); m++){
		
		//Get A submatrix
		subMatrix Asub(A, BLOCK_SIZE, blockRow, m);

		//Get B submatrix 
		subMatrix Bsub(B, BLOCK_SIZE, m ,blockCol);  
		

		//Load Asub and Bsub from global memory into shared; 

		As[row][col] = Asub.GetElem(row,col);
		Bs[row][col] = Bsub.GetElem(row,col); 

		//Always sync threads when loading shared memory before doing computation
		__syncthreads();

		//Multiply the submatrices
		for (int e = 0; e < BLOCK_SIZE; e++)
			Cvalue += As[row][e]*Bs[e][col];

		//synchronize to make sure all threads are done computing
		__syncthreads();
	}
	//write Csub back into global memory 
	//each thread writes one element
	Csub.SetElem(row, col, Cvalue);
}


void load_rand(Matrix w) {
	for(int i = 0; i < w.height; i++) {
		for(int j = 0; j < w.width; j++) {
			w.elements[i*w.stride + j] = (float) rand()/RAND_MAX ;
		}
	}
}

void dense(Matrix A, string activation, Matrix B, Matrix C)
{
    int Gpu = 1; 
    int toDev = 1, fromDev = 2; 

	load_rand(B);
    //Load A and B to device memory 
    //Allocate Matrix C
    Matrix d_A(A.width, A.height, A.stride, Gpu);
    Matrix d_B(B.width, B.height, B.stride, Gpu);
    Matrix d_C(C.width, C.height, C.stride, Gpu);
    d_A.load(A, toDev);
    d_B.load(B, toDev); 
	
    // Invoke Kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height/ dimBlock.y); 

	cout << "dimBlock: " << dimBlock.y << 'x' << dimBlock.x << endl; 
	cout << "dimGrid: " << dimGrid.y << 'x' << dimGrid.x << endl; 
    //Use HIP Events for timing
    hipEvent_t start, stop; 
    float time; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0); 

    matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&time, start, stop); 
    std::cout<< " Shared Memory Matrix Multiplication time =" << '\t' 
             << time << "ms" << std::endl; 

	// Read C from Device memory 
    C.load(d_C, fromDev);
	
    //Free device memory 
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





int main () {

	int Cpu = 0; 
	int N = 2; 
	int M = 2;
	int num = 4; 

	srand(SEED);


	Matrix A(N, M, 0, Cpu);
	Matrix W(num, A.width, 0,Cpu);
	Matrix B(num, A.height, 0, Cpu);
	

	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            A.elements[i*A.stride + j] = i+j;
    	}
	}

	cout << "A shape: " << A.height << 'x' << A.width << endl;
	for( int i = 0; i < A.height; i++) {
    	for( int j = 0; j < A.width; j++) {
            cout << A.elements[i*A.stride + j] << ' ';
    	}
		cout << endl;
	}

	// dense_serial( A, "leakyrelu",  W, B);
	dense(A, "", W, B); 
	cout << "B shape: " << B.height << 'x' << B.width << endl;
	for( int i = 0; i < B.height; i++) {
    	for( int j = 0; j < B.width; j++) {
            cout << B.elements[i*B.stride + j] << ' ';
    	}
		cout << endl;
	}

	cout << "W shape: " << W.height << 'x' << W.width <<endl;
	for( int i = 0; i < W.height; i++) {
    	for( int j = 0; j < W.width; j++) {
            cout << W.elements[i*W.stride + j] << ' ';
    	}
		cout << endl;
	}

	

}