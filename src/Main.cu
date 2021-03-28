#ifdef _WIN32
#include <WinSock2.h>
#include <Windows.h>
#include <stdint.h>

int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

#else
#include <sys/time.h>
#endif


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Feeder.h"
#include "DataFile.h"
#include "CudaWindow.h"
#include "CudaInput.h"
#include "InputVector.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>

#define BLOCK_SIZE 100
#define MAX_LAG 60
#define THREADS_PER_BLOCK 32

using namespace std::literals::chrono_literals;

template <int maxLag, int blockSize, typename Contained>
__global__ void autocorrelate(AutocorrelationCUDA::CudaWindow<maxLag, blockSize, Contained> window, int start, int* out);


int main() {

	//open file where data is stored
	std::unique_ptr<AutocorrelationCUDA::CudaInput<std::uint8_t>> dataFile = std::make_unique<AutocorrelationCUDA::InputVector<std::uint8_t>>("", "test1");
	
	//array in GPU memory to store output data
	int* out_d;
	cudaMalloc(&out_d, MAX_LAG * sizeof(int));

	//create circular array in GPU memory
	AutocorrelationCUDA::CudaWindow<MAX_LAG, BLOCK_SIZE, std::uint8_t> window{};

	int timesCalled; //counter
	dim3 numberOfBlocks = ceil((float)MAX_LAG / THREADS_PER_BLOCK); //number of blocks active on the GPU
	
	//timer
	AutocorrelationCUDA::Timer timer{[](std::vector<double> data){AutocorrelationCUDA::DataFile<double>::write(data, "out_timer.txt");},
									 [](){struct timeval tp;
									      gettimeofday(&tp, NULL);
									      return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);}};
	timer.start();
	for(timesCalled = 0; timesCalled < 16; ++timesCalled) {
		window.copyBlock(dataFile->read(BLOCK_SIZE), cudaMemcpyHostToDevice); //store in GPU memory one block of data
		autocorrelate <<< numberOfBlocks, THREADS_PER_BLOCK >>> (window, timesCalled * BLOCK_SIZE, out_d);
		cudaDeviceSynchronize();
		
		if(timesCalled == 9) {std::this_thread::sleep_for(100ms);}		

		timer.getInterval();
	}

	//copy output data from GPU to CPU
	std::vector<int> out(MAX_LAG);
	cudaMemcpy(out.data(), out_d, MAX_LAG * sizeof(int), cudaMemcpyDeviceToHost);

	window.clean(); //deallocates memory on GPU

	std::cout << timesCalled << "\n";
	for (int i = 0; i < MAX_LAG; ++i) {
		out[i] = out[i] / ((timesCalled * BLOCK_SIZE) - i);
		std::cout << i << " --> " << out[i] << std::endl;
	}

	//write output to file
	AutocorrelationCUDA::DataFile<int>::write(out);

}


template <int maxLag, int blockSize, typename Contained>
__global__ void autocorrelate(AutocorrelationCUDA::CudaWindow<maxLag, blockSize, Contained> window, int start, int* out) {
	if(threadIdx.x <= MAX_LAG){
		int absoluteThreadsIdx = blockIdx.x * blockDim.x + threadIdx.x;
		int partialSum = 0;
		for (int i = 0; i < BLOCK_SIZE; ++i) {
			if(i+start - absoluteThreadsIdx >= 0) {
				int a = window[i + start - absoluteThreadsIdx];
				int b = window[i + start];
				partialSum += a*b;
				//partialSum += window[i+start - threadIdx.x] * window[i+start];
			}
		}
		
		out[absoluteThreadsIdx] += partialSum;
	}
}
