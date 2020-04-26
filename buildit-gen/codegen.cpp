#include <iostream>
#include "builder/builder_context.h"
#include "builder/builder.h"
#include "blocks/c_code_generator.h"
#include "builder/static_var.h"
#include "builder/lib/utils.h"
#include <cnpy.h>
#include <cmath>

#include "constants.h"

using builder::static_var;
using builder::dyn_var;

typedef dyn_var<float*> array_t;
typedef dyn_var<float> float_d;
typedef dyn_var<int> int_t;
typedef static_var<int> int_s;


typedef std::vector<std::vector<std::vector<int>>> indices_t;
typedef dyn_var<float(float, float)> max_f_t;


void mm(array_t &BC, array_t &AC, int_t &threadidx, int_t &blockidx, int_t &blockidy, array_t &ACC, float *BA, float *bias, const int A_dim, const int B_dim, const int C_dim, const int A_blocks, const int C_blocks, const int Gy, const int Ny, const int * bounds, indices_t &Ny_indices, indices_t &B_indices, max_f_t &max_f) {	
	float_d RC = 0.0;
	int_t C_offset = blockidy * (C_dim / C_blocks);
	int_t groupId = threadidx / (Gsy);
	int_t lane = threadidx % (Gsy);

	int_s block = builder::up_cast_range(blockidx, A_blocks);
	int_s group = builder::up_cast_range(groupId, Gy);

	int_s A_offset = bounds[block];
	int_s block_NY = bounds[block+1] - A_offset;
	int_s old_b_idx = -1;
	for (int_s idx = 0; idx < Ny_indices[block][group].size(); idx++) {
		int_s ny_idx = Ny_indices[block][group][idx];
		int_s b_idx = B_indices[block][group][idx];
		if (b_idx != (int)old_b_idx) {
			RC = BC[b_idx * C_dim + C_offset + lane];
			old_b_idx = b_idx;
		}
		int_s a_idx = A_offset + (int)ny_idx;
		ACC[ny_idx] = RC * BA[b_idx * A_dim + a_idx];	
	}	
	for (int_s i = 0; i < (int)block_NY; i++) {
		AC[(A_offset + (int)i) * C_dim + C_offset + lane] = max_f(ACC[i] + bias[A_offset+(int)i], 0.0);
	}
}
int* load_balancer2(float *BA, const int B_dim, const int A_dim, int A_blocks, int &NY) {
	int total_nnz = 0;
	for (int i = 0; i < B_dim * A_dim; i++)
		if (std::abs(BA[i]) > EPS)
			total_nnz++;	
	float nnz_per_block = (float)total_nnz / A_blocks;
	int sums[A_dim];
	int cumsums[A_dim];
	for (int x = 0; x < A_dim; x++) {
		sums[x] = 0;
		for (int y = 0; y < B_dim; y++)
			if (std::abs(BA[y * A_dim + x]) > EPS)
				sums[x]++;
		cumsums[x] = x ? cumsums[x-1] + sums[x]: sums[x];
	}	
	int * bounds = new int[A_blocks + 1];
	for (int x = 0; x < A_blocks; x++) 
		for (bounds[x] = 0; bounds[x] < A_dim; bounds[x]++) 
			if (cumsums[bounds[x]] > nnz_per_block * x) 
				break;
	bounds[A_blocks] = A_dim;
	NY = -1;
	for (int i = 1; i < A_blocks + 1; i++) {
		int diff = bounds[i] - bounds[i-1];
		if (diff > NY)
			NY = diff;
	}
	return bounds;		
}

void get_idx_balanced (int * bounds, float * BA, int Gy, int A_dim, int B_dim, int A_blocks, indices_t &Ny_indices, indices_t &B_indices) {	
	Ny_indices.resize(A_blocks);
	B_indices.resize(A_blocks);
	for (int block = 0; block < A_blocks; block++) {
		Ny_indices[block].resize(Gy);
		B_indices[block].resize(Gy);
		int A_offset = bounds[block];
		int block_NY = bounds[block+1] - A_offset;
		int nnz = 0;
		for (int x = A_offset; x < A_offset + block_NY; x++) {
			for (int y = 0; y < B_dim; y++) {
				if (std::abs(BA[y * A_dim + x]) > EPS) 
					nnz += 1;
			}
		}	
		int nnz_per_group = nnz / Gy;
		int curr_group = 0;
		int curr_nnz = 0;
		for (int B_idx = 0; B_idx < B_dim; B_idx++) {
			for (int ny = 0; ny < block_NY; ny++) {
				int A_idx = A_offset + ny;
				if (std::abs(BA[B_idx * A_dim + A_idx]) > EPS) {
					B_indices[block][curr_group].push_back(B_idx);
					Ny_indices[block][curr_group].push_back(ny);
					curr_nnz++;
					if (curr_nnz > nnz_per_group) {
						curr_group += 1;
						curr_nnz = 0;
					}
				}
			}
		}
	}	
		
}
int main(int argc, char* argv[]) {
	// argv[1] = A_dim
	// argv[2] = B_dim
	// argv[3] = C_dim
	// argv[4] = A_blocks
	// argv[5] = C_blocks
	// argv[6] = Gy
	// argv[7] = infile
	// argv[8] = bias_file
	// argv[9] = outfile
	// argv[10] = informat
	// argv[11] = outformat
	if (argc < 12) {
		printf("%s <A_dim> <B_dim> <C_dim> <A_blocks> <C_blocks> <Gy> <infile> <bias_file> <outfile> <informat> <outformat>\n", argv[0]);
		return -1;
	}	
	
	const int A_dim = atoi(argv[1]);
	const int B_dim = atoi(argv[2]);
	const int C_dim = atoi(argv[3]);
	
	const int A_blocks = atoi(argv[4]);
	const int C_blocks = atoi(argv[5]);
	
	const int Gy = atoi(argv[6]);
	
	std::string infile = argv[7];
	std::string bias_file = argv[8];
	std::string outfile = argv[9];
	
	std::string informat = argv[10];
	std::string outformat = argv[11];
	// For now we are dealing only with NCHW
	assert(informat == "NCHW" && outformat == "NCHW");
	
	// Load the infile and bias
	cnpy::NpyArray arr = cnpy::npy_load(infile);
	float * BA = arr.data<float>();
	assert(arr.word_size = sizeof(float));
	assert(arr.shape.size() == 2 && arr.shape[0] == B_dim && arr.shape[1] == A_dim);
	

	cnpy::NpyArray arr2 = cnpy::npy_load(bias_file);
	float *bias = arr2.data<float>();
	assert(arr2.word_size == sizeof(float));
	assert(arr2.shape.size() == 1 && arr2.shape[0] == A_dim);

	// Setup the builder context
	builder::builder_context context;

	// The variables to be used during runtime	
	dyn_var<float*> &BC = *(context.assume_variable<array_t>("BC"));
	dyn_var<float*> &AC = *(context.assume_variable<array_t>("AC"));
	// ACC has to be assumed for now, because variable typed arrays cannot 
	// be used as template arugments. Maybe fix this later with global variable addressing
	dyn_var<float*> &ACC = *(context.assume_variable<array_t>("ACC"));
	max_f_t &max_f = *(context.assume_variable<max_f_t>("max_f"));
	
	
	// CUDA specific runtime variables
	dyn_var<int> &blockidx = *(context.assume_variable<int_t>("blockIdx.x"));
	dyn_var<int> &blockidy = *(context.assume_variable<int_t>("blockIdx.y"));
	dyn_var<int> &threadidx = *(context.assume_variable<int_t>("threadIdx.y"));
	
	int Ny;
	int * bounds = load_balancer2(BA, B_dim, A_dim, A_blocks, Ny);
	indices_t Ny_indices, B_indices;
	get_idx_balanced(bounds, BA, Gy, A_dim, B_dim, A_blocks, Ny_indices, B_indices);
	auto ast = context.extract_ast_from_lambda([&] {
		mm(BC, AC, threadidx, blockidx, blockidy, ACC, BA, bias, A_dim, B_dim, C_dim, A_blocks, C_blocks, Gy, Ny, bounds, Ny_indices, B_indices, max_f);
	});			
	block::c_code_generator::generate_code(ast, std::cout, 0);	
	
}
