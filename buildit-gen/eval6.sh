A_blocks=$1
C_blocks=$2
Gy_i=$3
Gy_d=$4
echo $A_blocks $C_blocks $Gy_i $Gy_d
mkdir -p build/dir_$1_$2_$3_$4
./build/codegen 512 512 196 $A_blocks $C_blocks $Gy_i $Gy_d ../mobilenet/contraction_1x1_6.npy ../mobilenet/contraction_1x1_6_bias.npy build/testing.cu NCHW NCHW build/dir_$1_$2_$3_$4/gencode.inc build/dir_$1_$2_$3_$4/AB.dat
#/usr/local/cuda/bin/nvcc  -std=c++11 -I -Xcompiler "-w" -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=sm_61 --use_fast_math -Xptxas "-v -dlcm=ca" driver.cu -I build/dir_$1_$2_$3_$4 -o build/dir_$1_$2_$3_$4/run -I /data/scratch/ziheng/cnpy/ -L /data/scratch/ziheng/cnpy/build/ -lcnpy -O3 -lineinfo -g -G
/usr/local/cuda/bin/nvcc  -std=c++11 -I -Xcompiler "-w" -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=sm_61 --use_fast_math -Xptxas "-v -dlcm=ca" driver.cu -I build/dir_$1_$2_$3_$4 -o build/dir_$1_$2_$3_$4/run -I /data/scratch/ziheng/cnpy/ -L /data/scratch/ziheng/cnpy/build/ -lcnpy -O3 
echo ./build/dir_$1_$2_$3_$4/run ./build/dir_$1_$2_$3_$4/AB.dat ./build/BC.npy ../mobilenet/contraction_1x1_6_bias.npy build/ref.npy || exit
./build/dir_$1_$2_$3_$4/run ./build/dir_$1_$2_$3_$4/AB.dat ./build/BC.npy ../mobilenet/contraction_1x1_6_bias.npy build/ref.npy || exit

