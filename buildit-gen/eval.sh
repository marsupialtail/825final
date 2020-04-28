A_blocks=$1
C_blocks=$2
Gy_i=$3
Gy_d=$4
echo $A_blocks $C_blocks $Gy_i $Gy_d
LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/data/scratch/ziheng/cnpy/build ./build/codegen 128 64 3136 $A_blocks $C_blocks $Gy_i $Gy_d ../mobilenet/contraction_1x1_1.npy ../mobilenet/contraction_1x1_1_bias.npy build/testing.cu NCHW NCHW build/gencode.inc build/AB.dat
/usr/local/cuda/bin/nvcc driver.cu -I build -o build/run -I /data/scratch/ziheng/cnpy/ -L /data/scratch/ziheng/cnpy/build/ -lcnpy
LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/data/scratch/ziheng/cnpy/build ./build/run ./build/AB.dat ./build/BC.npy ../mobilenet/contraction_1x1_1_bias.npy build/ref.npy || exit

