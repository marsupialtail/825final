#!/bin/bash

A_dim=128
B_dim=64
C_dim=3136
layer_num=1
infile=mobilenet/contraction_1x1_${layer_num}_transposed.npy
biasfile=mobilenet/contraction_1x1_${layer_num}_bias.npy

python make_BC.py $infile $C_dim NCHW NCHW $biasfile

for A_blocks in 1 2 4; do
	for C_blocks in 49 98; do
		for Gy in 1; do

			python  code_gen.py --A_dim $A_dim --B_dim $B_dim --C_dim $C_dim --A_blocks $A_blocks --C_blocks $C_blocks --Gy $Gy --infile $infile --fuse --infile_bias $biasfile --outfile testing.cu > ptx
			nvcc -arch=sm_61 -O3 -o test testing.cu --std=c++11 --compiler-options="-fsingle-precision-constant" -I/data/scratch/ziheng/zlib-1.2.11 -I/data/scratch/ziheng/cuda-10.1/include -I/data/scratch/ziheng/cnpy -L/data/scratch/ziheng/cnpy/build -L/data/scratch/ziheng/cuda-10.1/lib64 -lcnpy 
			./test
			#python code_gen_ptx.py --A_dim $A_dim --B_dim $B_dim --C_dim $C_dim --A_blocks $A_blocks --C_blocks $C_blocks --Gy $Gy --infile $infile --outfile testing.ptx --fuse --infile_bias $biasfile
			#ptxas -arch=sm_61 testing.ptx -o testing.cubin
			#sed -i "s/sm_61/sm_53/g" testing.ptx
			#ptxas -arch=sm_53 testing.ptx -o jetson-kernels/layer${layer_num}_${A_blocks}_${C_blocks}.cubin
                        #nvcc driver_spmm.cpp -w -O3 -I/data/scratch/ziheng/zlib-1.2.11 -I/data/scratch/ziheng/cuda-10.1/include -I/data/scratch/ziheng/cnpy -L/data/scratch/ziheng/cnpy/build -L/data/scratch/ziheng/cuda-10.1/lib64 -DA_dim=$A_dim,B_dim=$B_dim,C_dim=$C_dim,A_Blocks=$A_blocks,C_Blocks=$C_blocks,Gy=$Gy -lcuda -lcudart -lcnpy -o exe --std=c++11 -Xptxas="-v"
			#./exe
		done
	done
done
