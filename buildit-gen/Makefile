$(shell mkdir -p build)

all: build/codegen

build/codegen: codegen.cpp constants.h
	g++ codegen.cpp -I /data/scratch/ajaybr/buildit/include/ -lbuildit -L /data/scratch/ajaybr/buildit/build/ -I /data/scratch/ziheng/cnpy -L /data/scratch/ziheng/cnpy/build -lcnpy -o build/codegen -rdynamic -O3

clean:
	rm -rf build


run: build/codegen
	./build/codegen 128 64 3136 1 98 1 ../mobilenet/contraction_1x1_1_transposed.npy ../mobilenet/contraction_1x1_1_bias.npy build/testing.cu NCHW NCHW	
