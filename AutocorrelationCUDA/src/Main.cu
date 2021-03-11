#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Feeder.h"
#include "DataFile.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#define BLOCK_SIZE 20
#define MAX_LAG 10

using namespace std::chrono_literals;

__global__ void autocorrelate(std::uint8_t* blockStart, int blockLength, int maxLag, int* out);


int main() {
	
	AutocorrelationCUDA::DataFile<std::uint8_t> dataFile{"C:\\", "test1"};
	std::vector<std::uint8_t> data = dataFile.read(BLOCK_SIZE);

	std::uint8_t* data_d;
	cudaMalloc(&data_d, BLOCK_SIZE * sizeof(std::uint8_t));
	cudaMemcpy(data_d, data.data(), BLOCK_SIZE * sizeof(std::uint8_t), cudaMemcpyHostToDevice);

	int* out_d;
	cudaMalloc(&out_d, BLOCK_SIZE * sizeof(int));

	autocorrelate <<< 1, MAX_LAG >>> (data_d, BLOCK_SIZE, MAX_LAG, out_d);

	std::vector<int> out(MAX_LAG);
	cudaMemcpy(out.data(), out_d, MAX_LAG * sizeof(int), cudaMemcpyDeviceToHost);

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


__global__ void autocorrelate(std::uint8_t* blockStart, int blockLength, int maxLag, int* out) {

	int partialSum = 0;
	if(threadIdx.x <= maxLag){
		for (int i = 0; i < blockLength; ++i) {
			if(threadIdx.x+i < blockLength){
				partialSum += blockStart[i] * blockStart[threadIdx.x + i];
			}
		}
		out[threadIdx.x] = partialSum;
	}

}