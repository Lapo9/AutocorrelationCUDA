#ifndef AUTOCORRELATIONCUDA_WINDOW
#define AUTOCORRELATIONCUDA_WINDOW

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>

namespace AutocorrelationCUDA {


//A class that creates a circular array on the GPU and a method to fill in the array from CPU
template<int maxLag, int blockSize, typename Contained>
class CudaWindow final {
	
	public:

	//Allocates the array on the GPU.
	//The array is divided into blocks, CPU should add data to the array in amounts that match the block size.
	//The array on the GPU is designed to store as many blocks as necessary, based on the maxLag parameter (look at Algorithm_visualization.xlsm to understand the formula).
	__host__ CudaWindow() {
		size = (ceil((float)(maxLag - 1) / blockSize) + 1) * blockSize;
		cudaMalloc(&arr, size * sizeof(Contained));
	}


	//Returns the element at position i%size (always positive)
	__device__ Contained operator[](int i) {
		return arr[(i%size+size)%size];
	}
		
	
	//Copies the vector into the array on GPU memory. Vector size should be the same as blockSize, if not there will be holes in the array.
	__host__ int copyBlock(const std::vector<Contained>& source, cudaMemcpyKind direction) {
		cudaMemcpy(arr + (currentBlock * blockSize), source.data(), source.size() * sizeof(Contained), direction);
		currentBlock = (++currentBlock) % blocksCount();
		return currentBlock;
	}


	private:

	//Number of blocks in the array
	__device__ __host__ int blocksCount() {
		return ceil((float)(maxLag - 1) / blockSize) + 1;
	}

	Contained* arr;
	int currentBlock = 0;
	int size;
};

}

#endif 
