#ifndef AUTOCORRELATIONCUDA_WINDOW
#define AUTOCORRELATIONCUDA_WINDOW

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <iostream>

namespace AutocorrelationCUDA {


//A class that creates a circular array on the GPU and a method to fill in the array from CPU
template<typename Contained>
class CudaWindow final {
	
	public:

	//Allocates the array on the GPU.
	//The array is divided into blocks, CPU should add data to the array in amounts that match the block size.
	//The array on the GPU is designed to store as many blocks as necessary, based on the maxLag parameter (look at Algorithm_visualization.xlsm to understand the formula).
	__host__ CudaWindow(std::uint_fast32_t maxLag, std::uint_fast32_t blockSize) : blockSize{blockSize}, maxLag{maxLag}{
		size = (ceil((float)(maxLag - 1) / blockSize) + 1) * blockSize;
		size = pow(2, ceil(log2(size))); //if size is power of 2, modulus operation is WAY faster
		cudaMalloc(&arr, size * sizeof(Contained));
	}


	//Returns the element at position i%size (always positive)
	__device__ Contained operator[](std::uint_fast32_t i) {
		return arr[i&(size-1)]; //arr[i%size]
	}
		
	
	//Copies the vector into the array on GPU memory. Vector size should be the same as blockSize, if not there will be holes in the array.
	__host__ std::uint_fast32_t copyBlock(const std::vector<Contained>& source, cudaMemcpyKind direction) {
		cudaMemcpy(arr + (currentBlock * blockSize), source.data(), source.size() * sizeof(Contained), direction);
		currentBlock = (++currentBlock) % blocksCount();
		return currentBlock;
	}

	__host__ void clean() {
		cudaFree(arr);
	}


	private:

	//Number of blocks in the array
	__device__ __host__ std::uint_fast32_t blocksCount() {
		return ceil((float)(maxLag - 1) / blockSize) + 1;
	}

	Contained* arr;
	std::uint_fast32_t currentBlock = 0;
	std::uint_fast32_t size;
	const std::uint_fast32_t blockSize;
	const std::uint_fast32_t maxLag;

};

}

#endif 
