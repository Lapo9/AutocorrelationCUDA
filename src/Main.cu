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
#include "DataFile.h"
#include "CudaWindow.h"
#include "CudaInput.h"
#include "InputVector.h"
#include <iostream>
#include <vector>
#include <memory>

#define BLOCK_SIZE_DEFAULT 3200000
#define MAX_LAG_DEFAULT 1000
#define THREADS_PER_BLOCK_DEFAULT 256
#define REPETITIONS_DEFAULT 1000


template <typename Contained>
__global__ void autocorrelate(AutocorrelationCUDA::CudaWindow<Contained> window, std::uint_fast32_t start, std::uint_fast32_t maxLag, std::uint_fast32_t blockSize, std::uint_fast32_t* out);



std::vector<std::uint_fast32_t> askParameters() {
	std::vector<std::uint_fast32_t> result;
	std::uint_fast32_t tmp;

	std::cout << "\nInsert 0 for default parameter\n";
	std::cout << "\nblock size: ";			std::cin >> tmp;	result.emplace_back(tmp <= 0 ? BLOCK_SIZE_DEFAULT : tmp);
	std::cout << "\nmax lag: ";				std::cin >> tmp;	result.emplace_back(tmp <= 0 ? MAX_LAG_DEFAULT : tmp);
	std::cout << "\nthreads per block: ";	std::cin >> tmp;	result.emplace_back(tmp <= 0 ? THREADS_PER_BLOCK_DEFAULT : tmp);
	std::cout << "\nrepetitions: ";			std::cin >> tmp;	result.emplace_back(tmp <= 0 ? REPETITIONS_DEFAULT : tmp);

	return result;
}



int main() {
	
	//ask parameters to user
	std::vector<std::uint_fast32_t> params = askParameters();
	const std::uint_fast32_t blockSize = params[0];
	const std::uint_fast32_t maxLag = params[1];
	const std::uint_fast32_t threadsPerBlock = params[2];
	const std::uint_fast32_t repetitions = params[3];

	//open file where data is stored
	std::unique_ptr<AutocorrelationCUDA::CudaInput<std::uint8_t>> dataFile = std::make_unique<AutocorrelationCUDA::InputVector<std::uint8_t>>("", "test1");
	
	//array in GPU memory to store output data
	std::uint_fast32_t* out_d;
	cudaMalloc(&out_d, maxLag * sizeof(std::uint_fast32_t));

	//create circular array in GPU memory
	AutocorrelationCUDA::CudaWindow<std::uint8_t> window{maxLag, blockSize};

	std::uint_fast32_t timesCalled; //counter
	dim3 numberOfBlocks = ceil((float)maxLag / threadsPerBlock); //number of blocks active on the GPU
	
	//timer
	AutocorrelationCUDA::Timer timer{[](std::vector<double> data){AutocorrelationCUDA::DataFile<double>::write(data, "out_timer.txt");},
									 [](){struct timeval tp;
									      gettimeofday(&tp, NULL);
									      return ((double)tp.tv_sec + (double)tp.tv_usec * 0.000001);}};
	
	timer.start();
	for(timesCalled = 0; timesCalled < repetitions; ++timesCalled) {
		window.copyBlock(dataFile->read(blockSize), cudaMemcpyHostToDevice); //store in GPU memory one block of data
		cudaDeviceSynchronize();
		timer.getInterval();
		autocorrelate <<< numberOfBlocks, threadsPerBlock >>> (window, timesCalled * blockSize, maxLag, blockSize, out_d);
		cudaDeviceSynchronize();	
		timer.getInterval();
	}

	//copy output data from GPU to CPU
	std::vector<std::uint_fast32_t> out(maxLag);
	cudaMemcpy(out.data(), out_d, maxLag * sizeof(std::uint_fast32_t), cudaMemcpyDeviceToHost);

	window.clean(); //deallocates memory on GPU

	std::cout << timesCalled << "\n";
	for (std::uint_fast32_t i = 0; i < maxLag; ++i) {
		out[i] = out[i] / ((timesCalled * blockSize) - i);
		std::cout << i << " --> " << out[i] << std::endl;
	}

	//write output to file
	AutocorrelationCUDA::DataFile<std::uint_fast32_t>::write(out);

}


template <typename Contained>
__global__ void autocorrelate(AutocorrelationCUDA::CudaWindow<Contained> window, std::uint_fast32_t start, std::uint_fast32_t maxLag, std::uint_fast32_t dataSize, std::uint_fast32_t* out) {
	std::uint_fast32_t absoluteThreadsIdx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//outer cycle -> cycle over all of the new data
	for (int i = 0; i < dataSize; ++i) {
		//if(i % 2^absoluteThreadIdx) --> 
		//		zeroDelay[absoluteThreadsIdx+1] += zeroDelay[absoluteThreadsIdx] (maybe possible in parallel if we hold a local copy of zeroDelay where we can write to, and read from global zeroDelay)
		//		autocorrelate (parallel)
		//		shift
		//		accumulate
	}
}
