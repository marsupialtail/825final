#!/bin/bash

A_dim=512
B_dim=512
C_dim=196
layer_num=6
infile=mobilenet/contraction_1x1_${layer_num}_transposed.npy
biasfile=mobilenet/contraction_1x1_${layer_num}_bias.npy

python make_BC.py $infile $C_dim NCHW NCHW $biasfile

for A_blocks in 8 16 32; do
	for C_blocks in 1 2 4 7; do
		for Gy in 1; do
			echo $A_blocks $C_blocks $Gy
			python  code_gen.py --A_dim $A_dim --B_dim $B_dim --C_dim $C_dim --A_blocks $A_blocks --C_blocks $C_blocks --Gy $Gy --Fx 1 --infile $infile --outfile testing.cu > ptx
			nvcc -arch=sm_61 -O3 -o test testing.cu --std=c++11 --compiler-options="-fsingle-precision-constant" -I/data/scratch/ziheng/zlib-1.2.11 -I/data/scratch/ziheng/cuda-10.1/include -I/data/scratch/ziheng/cnpy -L/data/scratch/ziheng/cnpy/build -L/data/scratch/ziheng/cuda-10.1/lib64 -lcnpy 
			./test
		done
	done
done
