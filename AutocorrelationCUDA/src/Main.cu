#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Feeder.h"
#include "DataFile.h"
#include "CudaWindow.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#define BLOCK_SIZE 4
#define MAX_LAG 10

using namespace std::chrono_literals;

__global__ void autocorrelate(AutocorrelationCUDA::CudaWindow<MAX_LAG, BLOCK_SIZE, std::uint8_t> window, int start, int* out);


int main() {
	
	//read file where data is stored
	AutocorrelationCUDA::DataFile<std::uint8_t> dataFile{"C:\\", "test1"};
	
	//array in GPU memory to store output data
	int* out_d;
	cudaMalloc(&out_d, MAX_LAG * sizeof(int));

	AutocorrelationCUDA::CudaWindow<MAX_LAG, BLOCK_SIZE, std::uint8_t> window{};

	//copy read data to GPU
	for(int i=0; i<2; ++i){
		window.copyBlock(dataFile.read(BLOCK_SIZE), cudaMemcpyHostToDevice);

		autocorrelate <<< 1, MAX_LAG >>> (window, i*BLOCK_SIZE, out_d);
	}

	//copy output data from GPU to CPU
	std::vector<int> out(MAX_LAG);
	cudaMemcpy(out.data(), out_d, MAX_LAG * sizeof(int), cudaMemcpyDeviceToHost);

	//write output to file
	dataFile.write<int>(out);

	for (int i = 0; i < MAX_LAG; ++i) {
		std::cout << out[i] << std::endl;
	}

	/*Feeder f1{2s, [] {std::cout << "Ciao!\n" << std::endl; }};
	f1.start();

	std::this_thread::sleep_for(4s);
	f1.pause();
	std::this_thread::sleep_for(6s);
	f1.resume();

	std::this_thread::sleep_for(10s); */
	//receive data
	//send data to GPU
	//launch kernel
	//loop

	//collect results

}


__global__ void autocorrelate(AutocorrelationCUDA::CudaWindow<MAX_LAG, BLOCK_SIZE, std::uint8_t> window, int start, int* out) {
	if(threadIdx.x <= MAX_LAG){

		int partialSum = 0;
		for (int i = 0; i < BLOCK_SIZE; ++i) {
			if(i+start - threadIdx.x >= 0){
				partialSum += window[i+start - threadIdx.x] * window[i+start];
			}
		}
		
		out[threadIdx.x] += partialSum;
	}
}