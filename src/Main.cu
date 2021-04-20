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
#include "ResultArray.h"
#include "DataFile.h"
#include "CudaInput.h"
#include "InputVector.h"
#include "BinGroupsMultiSensorMemory.h"
#include "SensorsDataPacket.h"
#include <iostream>
#include <vector>
#include <memory>

namespace AutocorrelationCUDA {


#define REPETITIONS 20


__global__ void autocorrelate(SensorsDataPacket packet, BinGroupsMultiSensorMemory binStructure, uint32 instantsProcessed, ResultArray out);



	template<typename Integer>
	__device__ static Integer repeatTimes(Integer x, std::int8_t bits) {
		Integer pow2 = 1;
		for (std::uint8_t i = 0; i < bits; ++i) {
			if ((x & pow2) != 0) {
				return i + 1;
			}
			pow2 = pow2 << 1;
		}
		return 0;
	}



int main() {

	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	//open file where data is stored
	//std::unique_ptr<AutocorrelationCUDA::CudaInput<int>> dataFile = std::make_unique<AutocorrelationCUDA::InputVector<int>>("", "test1");
	

	//create array where to put new data for the GPU
	SensorsDataPacket inputArray;

	//array in GPU of the bin groups structure
	BinGroupsMultiSensorMemory binStructure;

	//output array to store results in GPU
	auto out = binStructure.generateResultArray();


	dim3 numberOfBlocks = SENSORS / SENSORS_PER_BLOCK; //number of blocks active on the GPU
	dim3 threadsPerBlock {GROUP_SIZE, SENSORS_PER_BLOCK};
	
	//timer
	Timer timer{[](std::vector<double> data){DataFile<double>::write(data, "out_timer.txt");},
									 [](){struct timeval tp;
									      gettimeofday(&tp, NULL);
									      return ((double)tp.tv_sec + (double)tp.tv_usec * 0.000001);}};
	

	std::vector<std::uint_fast8_t> dataDebug(SENSORS * INSTANTS_PER_PACKET);
	for (int i = 0; i < SENSORS * INSTANTS_PER_PACKET; ++i) {
		dataDebug[i] = i % 10 +1;
	}

	std::uint_fast32_t timesCalled; //counter
	timer.start();
	for(timesCalled = 0; timesCalled < REPETITIONS; ++timesCalled) {
		inputArray.setNewDataPacket(dataDebug); //store in GPU memory a new block of data to be processed
		//inputArray.setNewDataPacket(dataFile->read(sensors * instantsPerPacket)); //store in GPU memory a new block of data to be processed
		//cudaDeviceSynchronize();
		//timer.getInterval();
		//timer.start();
		autocorrelate <<< numberOfBlocks, threadsPerBlock >>> (inputArray, binStructure, timesCalled * INSTANTS_PER_PACKET, out);
		cudaDeviceSynchronize();	
		timer.getInterval();
	}
	

	std::cout << timesCalled << "\n";
	for (int sensor = 0; sensor < out.getSensors(); ++sensor) {
		std::cout << "\n\n\t======= SENSOR " << sensor << " =======\n";

		for (int lag = 0; lag < out.getMaxLagv(); ++lag) {
			int curr = out.get(sensor, lag);
			//int div = (timesCalled*instantsPerPacket) - lag;
			//float print = (float) curr / div;
			std::cout << "\n\t" << lag+1 << " --> " << curr;
		}
	}

	//write output to file
	//AutocorrelationCUDA::DataFile<std::uint_fast32_t>::write(out);

	cudaDeviceReset();
	return 0;
}





__global__ void autocorrelate(SensorsDataPacket packet, BinGroupsMultiSensorMemory binStructure, uint32 instantsProcessed, ResultArray out) {
	
	//precondition: blockDim.x = groupSize, blockDim.y = sensorsPerBlock

	uint16 blockDimTot = blockDim.x * blockDim.y;
	uint16 absoluteY = threadIdx.y + blockIdx.x * blockDim.y;
	uint8 relativeID = threadIdx.x + threadIdx.y * blockDim.x; //not more than 256 threads per block (basically 8 sensors)

	//put data in shared memory
	//TODO probably it is better to load only the most used groups (0, 1, 2, 3), to increase occupancy
	__shared__ uint8 sharedMemory[SHARED_MEMORY_REQUIRED];
	uint8* data = sharedMemory;
	uint8* accumulatorsPos = &data[SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE];
	uint8* zeroDelays = &accumulatorsPos[SENSORS_PER_BLOCK * GROUP_SIZE];
	uint8* output = &zeroDelays[SENSORS_PER_BLOCK * GROUP_SIZE];

	//copy data
	for (int group = 0; group < GROUPS_PER_SENSOR; ++group) {
		data[relativeID + group * SENSORS_PER_BLOCK * GROUP_SIZE] = binStructure.get(absoluteY, group, threadIdx.x);
		output[relativeID + group * SENSORS_PER_BLOCK * GROUP_SIZE] = 0; //set the output to 0
	}

	//copy accumulatorsPos and zeroDelays (groupsNum <= groupSize)
	if (relativeID < GROUPS_PER_SENSOR * SENSORS_PER_BLOCK) {
		accumulatorsPos[relativeID] = binStructure.getAccumulatorRelativePos(absoluteY, threadIdx.x);
		zeroDelays[relativeID] = binStructure.getZeroDelay(absoluteY, threadIdx.x);
	}

	__syncthreads();



	//we only have e.g. 31 values per group + 1 accumulator. This means that in the output there will be "holes" at positions multiple of e.g. 32
	if(threadIdx.x != blockDim.x-1){
		//cycle over all of the new data, where i is the instant in time processed
		for (int i = 0; i < INSTANTS_PER_PACKET; ++i) {
			instantsProcessed++;

			//only one thread per sensor adds the new datum to the bin structure
			if (threadIdx.x == 0) {
				BinGroupsMultiSensorMemory::insertNew(threadIdx.y, packet.get(absoluteY, i), data);
			}
			__syncthreads();
			
			//calculate autocorrelation for that instant
			//Decides how many group to calculate, based on how many instants have been already processed (i.e. 1 instant-->0; 2-->0,1; 3-->0; 4-->0,1,2; 5-->0; 6-->0,1; ...)
			std::uint_fast32_t repeatTimes = AutocorrelationCUDA::repeatTimes(instantsProcessed, 32);
			for (std::uint_fast8_t j = 0; j < repeatTimes; ++j) {

				output[relativeID + blockDimTot * j] += BinGroupsMultiSensorMemory::getZeroDelay(threadIdx.y, j, data) * BinGroupsMultiSensorMemory::get(threadIdx.y, j, threadIdx.x, data);
				__syncthreads();

				//only one thread per sensor makes the shift
				if (relativeID < 8) {
					BinGroupsMultiSensorMemory::shift(relativeID, j, data);
				}
				__syncthreads();
			}
		}
	}

	//TODO copy shared binStructure to global

	//copy output to total output
	for (int i = 0; i < binStructure.groupsNum(); ++i) {
		out.addTo(absoluteY, threadIdx.x + blockDim.x * i, output[relativeID + blockDim.x * i]);
	}
}

}
