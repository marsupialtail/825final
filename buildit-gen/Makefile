$(shell mkdir -p build)

all: build/codegen

build/codegen: codegen.cpp constants.h matrix.h
	g++ codegen.cpp -I /data/scratch/ajaybr/buildit/include/ -lbuildit -L /data/scratch/ajaybr/buildit/build/ -I /data/scratch/ziheng/cnpy -L /data/scratch/ziheng/cnpy/build -lcnpy -o build/codegen -rdynamic -O3 -g

clean:
	rm -rf build

.PHONY: run
run: build/codegen driver.cu build/codegen matrix.h
	bash autotune.sh

