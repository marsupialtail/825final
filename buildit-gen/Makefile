$(shell mkdir -p build)

all: build/codegen

build/codegen: codegen.cpp constants.h matrix.h
	g++ codegen.cpp -I /data/commit/graphit/ajaybr/scratch/buildit/include/ -lbuildit -L /data/commit/graphit/ajaybr/scratch/buildit/build/ -I /data/scratch/ziheng/cnpy -L /data/scratch/ziheng/cnpy/build -lcnpy -o build/codegen -rdynamic -O3 -g

clean:
	rm -rf build/codegen

.PHONY: run
run: build/codegen driver.cu build/codegen matrix.h
	python make_BC.py ../mobilenet/contraction_1x1_1_transposed.npy 3136 NCHW NCHW ../mobilenet/contraction_1x1_1_bias.npy
	bash eval.sh 1 49 1 1 

