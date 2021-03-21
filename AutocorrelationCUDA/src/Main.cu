#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Feeder.h"
#include "DataFile.h"
#include "CudaWindow.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#define BLOCK_SIZE 10
#define MAX_LAG 30

using namespace std::chrono_literals;

template <int maxLag, int blockSize, typename Contained>
__global__ void autocorrelate(AutocorrelationCUDA::CudaWindow<maxLag, blockSize, Contained> window, int start, int* out);


int main() {
	
	//open file where data is stored
	AutocorrelationCUDA::DataFile<std::uint8_t> dataFile{"", "test1"};
	
	//array in GPU memory to store output data
	int* out_d;
	cudaMalloc(&out_d, MAX_LAG * sizeof(int));

	//create circular array in GPU memory
	AutocorrelationCUDA::CudaWindow<MAX_LAG, BLOCK_SIZE, std::uint8_t> window{};

	int timesCalled = 0; //counter

	for(timesCalled = 0; timesCalled < 16; ++timesCalled) {
		window.copyBlock(dataFile.read(BLOCK_SIZE), cudaMemcpyHostToDevice); //store in GPU memory one block of data
		autocorrelate <<< 1, MAX_LAG >>> (window, timesCalled * BLOCK_SIZE, out_d);
	}

	//copy output data from GPU to CPU
	std::vector<int> out(MAX_LAG);
	cudaMemcpy(out.data(), out_d, MAX_LAG * sizeof(int), cudaMemcpyDeviceToHost);

	window.clean(); //deallocates memory on GPU

	std::cout << timesCalled << "\n";
	for (int i = 0; i < MAX_LAG; ++i) {
		//out[i] = out[i] / ((timesCalled * BLOCK_SIZE) - i);
		std::cout << i << " --> " << out[i] << std::endl;
	}

	//write output to file
	dataFile.write<int>(out);

	

}


template <int maxLag, int blockSize, typename Contained>
__global__ void autocorrelate(AutocorrelationCUDA::CudaWindow<maxLag, blockSize, Contained> window, int start, int* out) {
	if(threadIdx.x <= MAX_LAG){

		int partialSum = 0;
		for (int i = 0; i < BLOCK_SIZE; ++i) {
			if(i+start - threadIdx.x >= 0) {
				int a = window[i + start - threadIdx.x];
				int b = window[i + start];
				partialSum += a*b;
				//partialSum += window[i+start - threadIdx.x] * window[i+start];
			}
		}
		
		out[threadIdx.x] += partialSum;
	}
}
