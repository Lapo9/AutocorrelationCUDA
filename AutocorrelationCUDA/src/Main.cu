#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Feeder.h"
#include "DataFile.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#define TEST_SIZE 20

using namespace std::chrono_literals;

__global__ void autocorrelate(int* blockStart, int blockLength, int maxLag);


int main() {
	
	AutocorrelationCUDA::DataFile<int> dataFile{"C:\\", "test1"};
	std::vector<int> data = dataFile.read(TEST_SIZE);

	int* data_d;
	cudaMalloc(&data_d, TEST_SIZE * sizeof(float));
	cudaMemcpy(data_d, data.data(), TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	autocorrelate <<< 1, TEST_SIZE >>> (data_d, TEST_SIZE, 10);

	cudaMemcpy(data.data(), data_d, TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	dataFile.write(data);

	for (int i = 0; i < TEST_SIZE; ++i) {
		std::cout << data[i] << std::endl;
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


__global__ void autocorrelate(int* blockStart, int blockLength, int maxLag) {

	int partialSum = 0;
	if(threadIdx.x <= maxLag){
		for (int i = 0; i < blockLength; ++i) {
			if(threadIdx.x+i < blockLength){
				partialSum += blockStart[i] * blockStart[threadIdx.x + i];
			}
		}
		blockStart[threadIdx.x] = partialSum;
	}

}