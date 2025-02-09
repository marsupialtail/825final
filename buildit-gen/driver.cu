#include <cnpy.h>
#include <iostream>
#include "matrix.h"
#include <cuda.h>
#include <vector>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cublas_v2.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
#include <sys/time.h>

//void __global__ mm(const float * __restrict__ BC, const sparse_matrix AB, const float * __restrict__ bias, float *AC, float *offsets)
#include "gencode.inc"


int main(int argc, char* argv[]) {
	// argv[1] = sparse_matrix.data
	// argv[2] = BC file name
	// argv[3] = bias file name
	// argv[4] = ref.npy
	sparse_matrix AB;
	load_matrix(AB, argv[1]);
	
	cnpy::NpyArray arr1 = cnpy::npy_load(argv[2]);	
	float * BC = arr1.data<float>();
	assert(arr1.word_size == sizeof(float));
	assert(arr1.shape.size() == 2 && arr1.shape[0] == B_dim && arr1.shape[1] == C_dim);
	
	cnpy::NpyArray arr4 = cnpy::npy_load(argv[3]);
        float * bias = arr4.data<float>();
        assert(arr4.word_size = sizeof(float));	
	assert(arr4.shape.size()==1 && arr4.shape[0] == A_dim);	

	cnpy::NpyArray arr2 = cnpy::npy_load(argv[4]);
        float * AC = arr2.data<float>();
	
	float *d_BC, *d_AC, *d_bias;
	int *d_offsets;	
	sparse_matrix AB_d;

	cudaMalloc((void**)&d_BC, B_dim * C_dim * sizeof(float));
	cudaMalloc((void**)&d_AC, A_dim * C_dim * sizeof(float));
	
	AB_d = AB;
	cudaMalloc((void**)&AB_d.row_val, AB_d.nnz * sizeof(int));
	cudaMalloc((void**)&AB_d.columns, (AB_d.num_columns+1) * sizeof(int));
	cudaMalloc((void**)&AB_d.values, AB_d.nnz * sizeof(float));
	cudaMalloc((void**)&d_offsets, offsets_size * sizeof(int));	
	cudaMalloc((void**)&d_bias, A_dim * sizeof(int));	


	cudaMalloc((void**)&AB_d.rows, (AB_d.num_rows + 1) * sizeof(int));
	cudaMalloc((void**)&AB_d.column_val, (AB_d.nnz) * sizeof(int));
	cudaMalloc((void**)&AB_d.values_column, (AB_d.nnz) * sizeof(float));
	
	cudaMemcpy(d_BC, BC, B_dim * C_dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, bias, A_dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(AB_d.row_val, AB.row_val, AB.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(AB_d.columns, AB.columns, (AB.num_columns+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(AB_d.values, AB.values, AB.nnz * sizeof(float), cudaMemcpyHostToDevice);


	cudaMemcpy(AB_d.rows, AB.rows, (AB.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(AB_d.column_val, AB.column_val, (AB.nnz) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(AB_d.values_column, AB.values_column, (AB.nnz) * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_offsets, offsets, offsets_size * sizeof(int), cudaMemcpyHostToDevice);
	
	float *result;
	result = new float[A_dim * C_dim];
	
	dim3 BS(A_blocks, C_blocks);
	//dim3 TS(2, C_dim/C_blocks);
	int cthreads = (C_dim/C_blocks + 31)/32;
	cthreads *= 32;
	dim3 TS(cthreads, 2);

	for (int i = 0; i < 1000; i++)	
		mm<<<BS, TS>>>(d_BC, AB_d, d_bias, d_AC);

	cudaError_t code = cudaGetLastError();
	if (code != 0)
		printf("%s\n", cudaGetErrorString(code));

	cudaDeviceSynchronize();
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaProfilerStart();
	cudaEventRecord(start);
	cudaEventSynchronize(stop);

	struct timeval start_time;
	struct timeval end_time;
	cudaDeviceSynchronize();
	gettimeofday(&start_time, NULL);	
	for(int i = 0;i < 1000;i ++){
		mm<<<BS, TS>>>(d_BC, AB_d, d_bias, d_AC);
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_time, NULL);	

	cudaEventRecord(stop);
	cudaProfilerStop();
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << "kernel used " << time / 1000.0 << std::endl;

	end_time.tv_sec -= start_time.tv_sec;
	end_time.tv_usec -= start_time.tv_usec;
	time = end_time.tv_sec + end_time.tv_usec/1e6;	

	std::cout << "kernel used " << time / 1000.0 * 1000.0 << std::endl;

	cudaMemcpy(result, d_AC, A_dim * C_dim * sizeof(float), cudaMemcpyDeviceToHost);
	float error = 0 ;
	for (int i = 0; i < A_dim * C_dim; i++) 
		error += std::abs(result[i] - AC[i]);
	//std::cout << result[0] << " " << result[1] << " " << result[2] << std::endl;
	//std::cout << AC[0] << " " << AC[1] << " " << AC[2] << std::endl;
	std::cout << error << std::endl;
	if (error > 0.01)
		return -1;
	return 0;
}
