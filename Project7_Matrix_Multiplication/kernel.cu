/************************************
* STUDENT: DAVID PARKS              *
* PROJECT: 7 - ARRAY MULTIPLICATION *
* DUE DATE: TUES 6/11/18            *
*************************************/


////////////////STUDENT CODE (1/2) ////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>


//MOST OPTIMAL TILE DIM FOUND USING CUDA QUERY METHODS
#define TILE_DIM 32


__global__ void matrix_Mul_Kernel(float* A, float* B, float* C, int m, 
	int k, int n)
{
	__shared__ float sub_Tile_A[TILE_DIM][TILE_DIM];
	__shared__ float sub_Tile_B[TILE_DIM][TILE_DIM];

	// Identify the row and column of the P element to work on
	int row = blockIdx.y * TILE_DIM + threadIdx.y;
	int col = blockIdx.x * TILE_DIM + threadIdx.x;

	float c_Value = 0.0;
	
	// Loop over the M and N tiles required to compute the P element
	for (int i = 0; i < (TILE_DIM + k - 1.0)/TILE_DIM; i++) 
	{
		// Collaborative loading of M and N tiles into shared memory
		
		//CHECKS row IS IN BOUNDS AND THAT THREAD X IN TILE IN A DOES NOT SURPASS k
		if ( (row < m) && ((i * TILE_DIM + threadIdx.x) < k) )
		{
			sub_Tile_A[threadIdx.y][threadIdx.x] = A[(row * k) + (i * TILE_DIM) + threadIdx.x];
		}
		else
		{
			sub_Tile_A[threadIdx.y][threadIdx.x] = 0.0;
		}

		//CHECKS col IS IN BOUNDS AND THAT THREAD Y IN TILE IN B DOES NOT SURPASS k
		if ( (col < n) && (((i * TILE_DIM) + threadIdx.y) < k) )
		{
			sub_Tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_DIM + threadIdx.y) * n + col];
		}
		else
		{
			sub_Tile_B[threadIdx.y][threadIdx.x] = 0.0;
		}

		//SYNC THREADS SO VALUES ARE NOT COMPUTED WITH MISSING TILE VALUES
		__syncthreads();

		//COMPUTE VALUE FOR THREAD IN C
		for (int j = 0; j < TILE_DIM; ++j)
		{
			c_Value += sub_Tile_A[threadIdx.y][j] * sub_Tile_B[j][threadIdx.x];
		}
		//SYNC THREADS SO VALUES ARE NOT CHANGED WHILE OTHERS THREADS COMPUTING c_Value
		__syncthreads();
	}
	//SET THREAD POSITION IN C TO c_Value
	if ( (row < m) && (col < n) )
		C[((blockIdx.y * blockDim.y + threadIdx.y) * n) +
		(blockIdx.x * blockDim.x) + threadIdx.x] = c_Value;
}

//////////////// END STUDENT CODE (1/2) ////////////////




//VERIFY METHOD TO CHECK CUDA FUNCTION - GIVEN BY PROFESSOR
void verify(float *A, float *B, float *C, int m, int k, int n) {
	const float relativeTolerance = 1e-6;
	for (int row = 0; row < m; ++row) {
		for (int col = 0; col < n; ++col) {
			float sum = 0;
			for (int i = 0; i < k; ++i) {
				sum += A[row*k + i] * B[i*n + col];
			}
			float relativeError = (sum - C[row*n + col]) / sum;
			if (relativeError > relativeTolerance ||
				relativeError < -relativeTolerance) {
				printf("\n\n\nTEST FAILED\n\n");
				exit(0);
			}
		}
	}
	printf("\n\n\nTEST PASSED\n\n");
}




//////////////// STUDENT CODE (2/2) ////////////////

//GENERATES RANDOM VALUES FOR MxN MATRIX WITH LOWER AND UPPER BOUNDS
void gen_Rand_Matrix(float* arr, int m, int n, int upper_Bound, int lower_Bound)
{
	//GIVE ALL VALUES 'RANDOM' FLOAT IN RANGE OF BOUNDS
	for (int i = 0; i < m*n; i++)
		arr[i] = lower_Bound + (rand() / ((float)RAND_MAX/(upper_Bound - lower_Bound)));

}


//PRINT ARRAYS OF MxN DIMENSIONS
void print_Array(float* arr, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%f, ", arr[i*n + j]);
		}
		printf("\n");
	}
}


//PRINT USAGE FOR COMMAND LINE
void print_Usage(char *file_Name, int max_Value, int min_Value)
{
	printf("Usage: %s m k n [s|p] [max_matrix_value] [min_matrix_value]\n"
		"(Matrix multiplication requires k to be the same value for both "
		"matricies.)\nAdding 's' as the 4th argument will disable "
		"printing the arrays for larger array dimensions.\n\tAdding a 'p' "
		"in place of the 's' will allow arrays to be printed.\n\tArrays are "
		"printed if left alone.\nA maximum and minimum value and be set "
		"for matrix values, however, when not given, the\n\tdefault values "
		"will be %d and %d.\n\n\n", file_Name, max_Value, min_Value);
}


int main(int argc, char *argv[])
{
	//MIN AND MAX FLOAT VALUES
	int max_Float = 100;
	int min_Float = -max_Float;

	//CHECK FOR CORRECT NUMBER OF ARGS
	if ( argc != 4 && argc != 5 && argc != 7 )
	{
		printf("ERROR: INCORRECT NUMBER OF ARGUMENTS\n\n");
		print_Usage(argv[0], max_Float, min_Float);
		return 1;
	}

	//PRINT ARRAYS CONTROL
	int show_Arrays = 1;

	//HANDLES INPUT FOR NOT PRINTING ARRAYS
	if (argc == 5)
	{
		if ( (int)(char)argv[4][0] == (int)'s' )
		{
			show_Arrays = 0;
		}
		else if ( (int)(char)argv[4][0] != (int)'p' )
		{
			printf("ERROR: UNEXPECTED 4TH ARGUMENT\n\n");
			print_Usage(argv[0], max_Float, min_Float);
			return 1;
		}
	}

	//HANDLES INPUT FOR MAX MIN FLOAT VALUES AND PRINTING ARRAYS
	if ( argc == 7 )
	{
		if ( (int)(char)argv[4][0] == (int)'s' )
		{
			show_Arrays = 0;
		}
		else if( (int)(char)argv[4][0] != (int)'p' )
		{
			printf("ERROR: UNEXPECTED 4TH ARGUMENT\n\n");
			print_Usage(argv[0], max_Float, min_Float);
			return 1;
		}

		if ( argv[5] > argv[6] )
		{
			max_Float = atoi(argv[5]);
			min_Float = atoi(argv[6]);
		}
		else if ( argv[5] < argv[6] )
		{
			max_Float = atoi(argv[6]);
			min_Float = atoi(argv[5]);
		}
		else
		{
			printf("ERROR: INVALID INPUT FOR MATRIX VALUE RANGE \n\t-"
				"The max and min values cannot be equal.\n");
			print_Usage(argv[0], max_Float, min_Float);
			return 1;
		}
	}


	//QUERY FOR DEVICE INFO
	int num_devices;
	cudaGetDeviceCount(&num_devices);

	cudaDeviceProp dev_Prop;

	for (int i = 0; i < num_devices; i++)
		cudaGetDeviceProperties(&dev_Prop, i);
	
	//CUDE DEVICE PRINT INFO
	printf("This machine has %d bytes per block of shared memory.\nAnd this machine also has a max amount "
		"of %d threads per block.\nWith this information, we can find that we can have a max size of tile "
		"width being around 78.\nSadly this would require more than the allowed number of threads. \nSo "
		"using a size of 32 for TILE_SIZE uses the max amount of threads per block and %d bytes of shared "
		"memory.\n\n\n", dev_Prop.sharedMemPerBlock, dev_Prop.maxThreadsPerBlock, 32 * 32 * sizeof(float));
	

	//INPUT PARAMETERS FOR MATRIX DEMSNTIONS M, K, M
	int m = atoi(argv[1]);
	int k = atoi(argv[2]);
	int n = atoi(argv[3]);


	//HOST AND DEVICE ARRAY FOR A, B, AND C (A*B=C)
	float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;


	//CUDA ERROR FOR ERROR INFO IN MALLOCS AND MEMCPY'S
	cudaError_t cudaStatus;


	//ALLOCATE MEMORY FOR HOST ARRAYS ACCORDING TO MATRIX DIMS.
	h_A = (float*)malloc(m * k * sizeof(float));
	if ( !h_A )
	{
		printf("Host memory allocation failed");
		free(h_A);
		return 1;
	}
	h_B = (float*)malloc(k * n * sizeof(float));
	if ( !h_B )
	{
		printf("Host memory allocation failed");
		free(h_A);
		free(h_B);
		return 1;
	}
	h_C = (float*)malloc(m * n * sizeof(float));
	if ( !h_C )
	{
		printf("Host memory allocation failed");
		free(h_A);
		free(h_B);
		free(h_C);
		return 1;
	}


	//allocate memory on GPU
	cudaStatus = cudaMalloc((void**)&d_A, m * k * sizeof(float));
	if ( cudaStatus != cudaSuccess )
	{
		printf("Could not allocate space on GPU");
		free(h_A);
		free(h_B);
		free(h_C);
		cudaFree(d_A);
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&d_B, k * n * sizeof(float));
	if ( cudaStatus != cudaSuccess )
	{
		printf("Could not allocate space on GPU");
		free(h_A);
		free(h_B);
		free(h_C);
		cudaFree(d_A);
		cudaFree(d_B);
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&d_C, m * n * sizeof(float));
	if ( cudaStatus != cudaSuccess )
	{
		printf("Could not allocate space on GPU");
		free(h_A);
		free(h_B);
		free(h_C);
		cudaFree(d_A);
		cudaFree(d_B); 
		cudaFree(d_C);
		return 1;
	}

	
	//SEED SRAND FOR RAND MATRICES
	srand(time(NULL));


	//FILL VALUES FOR HOST ARRAYS
	gen_Rand_Matrix(h_A, m, k, max_Float, min_Float);
	gen_Rand_Matrix(h_B, k, n, max_Float, min_Float);


	//copy memory from host to GPU
	cudaStatus = cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
	if ( cudaStatus != cudaSuccess )
	{
		printf("Could not copy memory from host to GPU");
		free(h_A);
		free(h_B);
		free(h_C);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		return 1;
	}

	cudaStatus = cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
	if ( cudaStatus != cudaSuccess )
	{
		printf("Could not copy memory from host to GPU");
		free(h_A);
		free(h_B);
		free(h_C);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		return 1;
	}


	//BLOCK DEMENSIONS 
	dim3 dimBlock(TILE_DIM, TILE_DIM);
	dim3 dimGrid((TILE_DIM + n - 1) / TILE_DIM, (TILE_DIM + m - 1)  / TILE_DIM);

	//KERNEL CALL
	matrix_Mul_Kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);

	
	//COPY C BACK FROM KERNEL AFTER COMPUTATION
	cudaStatus = cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	if ( cudaStatus != cudaSuccess )
	{
		printf("Could not copy memory from GPU to host");
		free(h_A);
		free(h_B);
		free(h_C);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		return 1;
	}
	

	//SHOW ARRAYS UNLESS 's' IS GIVEN AS 4TH ARG
	if ( show_Arrays )
	{
		printf("Array A: \n\n");
		print_Array(h_A, m, k);
		printf("\n\n\n");

		printf("Array B: \n\n");
		print_Array(h_B, k, n);
		printf("\n\n\n");

		printf("Array C: \n\n");
		print_Array(h_C, m, n);
	}
	

	//VERIFY TO CHECK KERNEL METHOD'S CORRECTNESS
	verify(h_A, h_B, h_C, m, k, n);


	//FREE MEMORY FOR HOST AND DEVICE
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
} 
//////////////// END STUDENT CODE (2/2) ////////////////