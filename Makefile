NVCC=nvcc
CXX=g++

CXXFLAGS=-Ofast -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES
LIBS=-lcudart
FLAGS=-Xcompiler -fopenmp -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES

CUDAFLAGS=--ptxas-options=-v -O4 -m64 -arch compute_61 -code sm_61 -Xptxas -dlcm=ca -Xcompiler -D_FORCE_INLINES 

all: main

clean:
	rm -f *.o main

main: src/CudaWindow.h  src/DataFile.h  src/Feeder.h src/Main.cu
	$(NVCC) $(CUDAFLAGS) src/Main.cu -o main

