#ifndef MATRIX_H
#define MATRIX_H
#include "constants.h"

struct sparse_matrix { // AB
	int32_t num_rows; // A
	int32_t num_columns; // B

	int32_t nnz;

	int32_t * rows; // size = outer dim (A_dim + 1)


	int32_t * columns; // size = inner dim(B_dim + 1)
	int32_t * row_val; // size = nnz;
	float * values; // size = nnz;

	float * dense; // size = A_dim * B_dim
};


static sparse_matrix to_sparse(int A_dim, int B_dim, float * AB) {
	int nnz = 0;
	for (int a_idx = 0; a_idx < A_dim; a_idx++) {
		for (int b_idx = 0; b_idx < B_dim; b_idx++) {
			if (std::abs(AB[a_idx * B_dim + b_idx]) > EPS) {
				nnz++;
			}
		}	
	}
	int *rows = new int[A_dim+1];
	int *columns = new int[B_dim+1];

	int *row_val = new int[nnz];
	float *values = new float[nnz];

	int rnnz = 0;
	rows[0] = 0;
	for (int a_idx = 0; a_idx < A_dim; a_idx++) {
		for (int b_idx = 0; b_idx < B_dim; b_idx++) {
			if (std::abs(AB[a_idx * B_dim + b_idx]) > EPS) {
				rnnz++;
			}
		}
		rows[a_idx+1] = rnnz;
	}
	rnnz = 0;
	// Now do the same for values and columns
	columns[0] = 0;
	for (int b_idx = 0; b_idx < B_dim; b_idx++) {
		for (int a_idx = 0; a_idx < A_dim; a_idx++) {
			if (std::abs(AB[a_idx * B_dim + b_idx]) > EPS) {
				row_val[rnnz]  = a_idx;
				values[rnnz] = AB[a_idx * B_dim + b_idx];
				rnnz++;
			}	
		}
		columns[b_idx+1] = rnnz;
	}
	sparse_matrix mtx;
	mtx.num_rows = A_dim;
	mtx.num_columns = B_dim;
	mtx.nnz = nnz;
	mtx.rows = rows;
	mtx.columns = columns;
	mtx.row_val = row_val;
	mtx.values = values;
	mtx.dense = AB;
	return mtx;
}
void save_matrix(sparse_matrix &mtx, std::string filename) {	
	FILE *bin_file = fopen(filename.c_str(), "wb");
	fwrite(&mtx.num_rows, sizeof(int), 1, bin_file);
	fwrite(&mtx.num_columns, sizeof(int), 1, bin_file);
	fwrite(&mtx.nnz, sizeof(int), 1, bin_file);
	fwrite(mtx.row_val, sizeof(int), mtx.nnz, bin_file);
	fwrite(mtx.values, sizeof(float), mtx.nnz, bin_file);
	fclose(bin_file);
}
void load_matrix(sparse_matrix &mtx, std::string filename) {	
	FILE *bin_file = fopen(filename.c_str(), "rb");
	assert(fread(&mtx.num_rows, sizeof(int), 1, bin_file) != 0);
	assert(fread(&mtx.num_columns, sizeof(int), 1, bin_file) != 0);
	assert(fread(&mtx.nnz, sizeof(int), 1, bin_file) != 0);
	mtx.row_val = new int[mtx.nnz];
	mtx.values = new float[mtx.nnz];
	assert(fread(mtx.row_val, sizeof(int), mtx.nnz, bin_file) != 0);
	assert(fread(mtx.values, sizeof(float), mtx.nnz, bin_file) != 0);
	fclose(bin_file);
}
#endif
