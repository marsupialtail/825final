#include <iostream>
#include <fstream>
#include "builder/builder_context.h"
#include "builder/builder.h"
#include "blocks/c_code_generator.h"
#include "builder/static_var.h"
#include "builder/lib/utils.h"
#include <cnpy.h>
#include <cmath>


#include "constants.h"
#include "matrix.h"

using builder::static_var;
using builder::dyn_var;

typedef dyn_var<float*> array_t;
typedef dyn_var<float> float_d;
typedef dyn_var<int> int_t;
typedef static_var<int> int_s;


typedef dyn_var<float(float, float)> max_f_t;

// mm(BC, AC, threadidx, threadidy, blockidx, blockidy, ACC, BA, bias, A_dim, B_dim, C_dim, A_blocks, C_blocks, Gy_i, Gy_d, bounds, max_f, BA_d, bias_d);
void mm(array_t &BC, array_t &AC, int_t &threadidx, int_t &threadidy, int_t &blockidx, int_t &blockidy, array_t &ACC, const sparse_matrix &AB, const float * bias, const int A_dim, const int B_dim, const int C_dim, const int A_blocks, const int C_blocks, const int Gy_i, const int Gy_d, std::vector<int> &bounds, const int* offsets, max_f_t &max_f, array_t &AB_values_d, dyn_var<int*> &AB_row_val_d, dyn_var<int*> &offsets_d, array_t &bias_d) {
	float_d RC = 0.0;
	int_t c_index = blockidy * (C_dim/C_blocks) + threadidy;
	int_t a_index = blockidx * (Gy_i + Gy_d) + threadidx;
	
	// Now we promote the a_index
	int_s a_idx = builder::up_cast_range(a_index, A_blocks * (Gy_i + Gy_d));
	
	int_s start_a = bounds[a_idx];
	int_s end_a = bounds[a_idx+1];


	if ((a_idx % (Gy_i + Gy_d)) < Gy_i) {
		// iterate over the sparse matrix

		for (int_s b_idx = 0; b_idx < B_dim; b_idx++) {
			// Find the start and end
			if (offsets[2*(a_idx * B_dim + b_idx)] != -1) {
				int_s start = offsets[2 * (a_idx * B_dim + b_idx)];
				int_s end = offsets[2 * (a_idx * B_dim + b_idx) + 1];
				RC = BC[b_idx * C_dim + c_index];
				for (int_s a_it = (int)start; a_it < (int)end; a_it++) {
					int_s a_val = AB.row_val[a_it];
					ACC[a_val - (int) start_a] = ACC[a_val - (int) start_a] + RC * AB.values[a_it];
				}
			}
		}	
		int_s a_it;
		for (a_it = start_a; a_it < (int)end_a; a_it++) {
			AC[a_it * C_dim + c_index] = max_f(ACC[(a_it - (int)start_a)] + bias[a_it], 0.0);
		}
	} else {
		// Do the same just use the dynamic loops this time
		for (int_t b_idx = 0; b_idx < B_dim; b_idx = b_idx + 1) {
			RC = BC[b_idx * C_dim + c_index];
			int_t start = offsets_d[2 * (a_idx * B_dim + b_idx)];
			int_t end = offsets_d[2 * (a_idx * B_dim + b_idx) + 1];
			for (int_t a_it = start; a_it < end; a_it = a_it + 1) {
				int_t a_val = AB_row_val_d[a_it];
				ACC[a_val - start_a] = ACC[a_val - start_a] + RC * AB_values_d[a_it];
			}
		}
		for (int_t a_it = start_a; a_it < (int)end_a; a_it = a_it + 1) {
			AC[a_it * C_dim + c_index] = max_f(ACC[(a_it - start_a)] + bias_d[a_it], 0.0);
		}
	}

	
}

std::vector<int> divide_A(sparse_matrix &AB, const int A_threads, int &max_bound) {
	int nnz = AB.nnz;
	int nnz_per_thread = (nnz+A_threads-1) / A_threads;
	std::vector<int> bounds;
	int thread_id = 0;
	bounds.push_back(0);
	for (int a_idx = 0; a_idx < AB.num_rows; a_idx++) {
		if (AB.rows[a_idx] >= nnz_per_thread * (thread_id+1)) {
			bounds.push_back(a_idx);
			thread_id++;
		}
	}
	bounds.push_back(AB.num_rows);
	for (auto a: bounds) {
		std::cout << a << ", ";
	}
	std::cout << std::endl;
	max_bound = -1;
	for (int i = 0; i < bounds.size() - 1; i++) {
		int diff = bounds[i+1] - bounds[i];
		if (diff > max_bound)
			max_bound = diff;
	}
	
	return bounds;
}

int* compute_offsets(sparse_matrix &AB, const int A_threads, const std::vector<int> &bounds) {
	int B_dim = AB.num_columns;
	int *offsets = new int [A_threads * B_dim * 2];
	for (int a_idx = 0; a_idx < A_threads; a_idx++) {
		int start_a = bounds[a_idx];
		int end_a = bounds[a_idx+1];
		for (int b_idx = 0; b_idx < B_dim; b_idx++) {
			// Find the start and end
			int start = -1;
			int end = AB.columns[b_idx+1];
			for (int a_it = AB.columns[b_idx]; a_it < AB.columns[b_idx+1]; a_it++) {
				int a_val = AB.row_val[a_it];
				if (a_val >= (int)start_a) {
					start = a_it;
					break;
				}
			}
			for (int a_it = AB.columns[b_idx]; a_it < AB.columns[b_idx+1]; a_it++) {
				int a_val = AB.row_val[a_it];
				if (a_val >= (int)end_a) {
					end = a_it;	
					break;
				}
			}
			if (start == -1)
				end = -1;
			offsets[2*(a_idx * B_dim + b_idx)] = start;
			offsets[2*(a_idx * B_dim + b_idx)+1] = end;
		}	
	}
	return offsets;
}

int main(int argc, char* argv[]) {
	// argv[1] = A_dim
	// argv[2] = B_dim
	// argv[3] = C_dim
	// argv[4] = A_blocks
	// argv[5] = C_blocks
	// argv[6] = Gy_i
	// argv[7] = Gy_d
	// argv[8] = infile
	// argv[9] = bias_file
	// argv[10] = outfile
	// argv[11] = informat
	// argv[12] = outformat
	// argv[13] = outfile
	// argv[14] = outfile_AB
	if (argc < 15) {
		printf("%s <A_dim> <B_dim> <C_dim> <A_blocks> <C_blocks> <Gy_i> <Gy_d> <infile> <bias_file> <outfile> <informat> <outformat> <genfile> <genfile_AB>\n", argv[0]);
		return -1;
	}	
	
	const int A_dim = atoi(argv[1]);
	const int B_dim = atoi(argv[2]);
	const int C_dim = atoi(argv[3]);
	
	const int A_blocks = atoi(argv[4]);
	const int C_blocks = atoi(argv[5]);
	
	const int Gy_i = atoi(argv[6]);
	const int Gy_d = atoi(argv[7]);
	
	std::string infile = argv[8];
	std::string bias_file = argv[9];
	std::string outfile = argv[10];
	
	std::string informat = argv[11];
	std::string outformat = argv[12];
	// For now we are dealing only with NCHW
	assert(informat == "NCHW" && outformat == "NCHW");
	
	// Load the infile and bias
	cnpy::NpyArray arr = cnpy::npy_load(infile);
	float * AB_dense = arr.data<float>();
	assert(arr.word_size = sizeof(float));
	assert(arr.shape.size() == 2 && arr.shape[0] == A_dim && arr.shape[1] == B_dim);
	

	cnpy::NpyArray arr2 = cnpy::npy_load(bias_file);
	float *bias = arr2.data<float>();
	assert(arr2.word_size == sizeof(float));
	assert(arr2.shape.size() == 1 && arr2.shape[0] == A_dim);

	// Before we do anything, let us construct a sparse matrix 
	// with EPS
	sparse_matrix AB = to_sparse(A_dim, B_dim, AB_dense); 	

	// We will first now divide A_dim equally among A_blocks 
	// So that each block gets almost equal number of nnzs to process
	int max_bound = 0;
	std::vector<int> bounds = divide_A(AB, A_blocks * (Gy_i + Gy_d), max_bound);
	
	// Now calculate the start and end for each group
	int * offsets = compute_offsets(AB, A_blocks * (Gy_i + Gy_d), bounds);
		

	// Setup the builder context
	builder::builder_context context;

	// The variables to be used during runtime	
	dyn_var<float*> &BC = *(context.assume_variable<array_t>("BC"));
	dyn_var<float*> &AC = *(context.assume_variable<array_t>("AC"));
	dyn_var<float*> &BA_d = *(context.assume_variable<array_t>("BA"));
	dyn_var<float*> &bias_d = *(context.assume_variable<array_t>("bias"));
	dyn_var<int*> &AB_row_val_d = *(context.assume_variable<dyn_var<int*>>("AB.row_val"));
	dyn_var<float*> &AB_values_d = *(context.assume_variable<array_t>("AB.values"));
	dyn_var<int*> &offsets_d = *(context.assume_variable<dyn_var<int*>>("offsets"));
	// ACC has to be assumed for now, because variable typed arrays cannot 
	// be used as template arugments. Maybe fix this later with global variable addressing
	dyn_var<float*> &ACC = *(context.assume_variable<array_t>("ACC"));
	max_f_t &max_f = *(context.assume_variable<max_f_t>("max_f"));
	
	
	// CUDA specific runtime variables
	dyn_var<int> &blockidx = *(context.assume_variable<int_t>("blockIdx.x"));
	dyn_var<int> &blockidy = *(context.assume_variable<int_t>("blockIdx.y"));
	dyn_var<int> &threadidx = *(context.assume_variable<int_t>("threadIdx.x"));
	dyn_var<int> &threadidy = *(context.assume_variable<int_t>("threadIdx.y"));
	

	// A_blocks = gridDim.x
	// C_blocks = gridDim.y
	
	// Gy_i + Gy_d = blockDim.x
	// each thread in x has to handle a few A's which will be unrolled
	
	// C_dim / C_blocks = blockDim.y
	// Thus total threads in y dim will be equal = C_dim
	// This way we don't have to loop over the C dimension at all
	
	
	save_matrix(AB, argv[14]);
	

	auto ast = context.extract_ast_from_lambda([&] {
		mm(BC, AC, threadidx, threadidy, blockidx, blockidy, ACC, AB, bias, A_dim, B_dim, C_dim, A_blocks, C_blocks, Gy_i, Gy_d, bounds, offsets, max_f, AB_values_d, AB_row_val_d, offsets_d, bias_d);
	});			
	
    	std::ofstream output_file;
    	output_file.open(argv[13]);

	std::ostream &oss(output_file);	
	oss << "#define A_dim (" << A_dim << ")" << std::endl;
	oss << "#define B_dim (" << B_dim << ")" << std::endl;
	oss << "#define C_dim (" << C_dim << ")" << std::endl;
	oss << "#define Gy_i (" << Gy_i << ")" << std::endl;
	oss << "#define Gy_d (" << Gy_d << ")" << std::endl;
	oss << "#define A_blocks (" << A_blocks << ")" << std::endl;
	oss << "#define C_blocks (" << C_blocks << ")" << std::endl;
	
	oss << "#define offsets_size (" << A_blocks * (Gy_i + Gy_d) * B_dim * 2 << ")" << std::endl;
	oss << "float __device__ max_f(float a, float b) {return a>b?a:b;}" << std::endl;
	oss << "static int offsets[] = {";
	for (int i = 0; i < A_blocks * (Gy_i + Gy_d) * B_dim * 2; i++) {
		oss << offsets[i] << ", ";
	}
	oss << "};" << std::endl;
	
	oss << "void __global__ mm(const float * __restrict__ BC, const sparse_matrix AB, const float * __restrict__ bias, float *AC, int *offsets) {" << std::endl;
	oss << "  register float ACC[" << max_bound << "] = {0.0}; " << std::endl;
	block::c_code_generator::generate_code(ast, oss, 1);	
	oss << "}" << std::endl;
	output_file.close();
	
}
