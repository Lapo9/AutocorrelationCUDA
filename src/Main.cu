﻿#ifdef _WIN32
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
#include <iostream>
#include <vector>
#include <memory>

#include "Definitions.h"
#include "ResultArray.h"
#include "BinGroupsMultiSensorMemory.h"
#include "SensorsDataPacket.h"

#include "DataFile.h"
#include "Timer.h"



using namespace AutocorrelationCUDA;



__global__ void autocorrelate(SensorsDataPacket packet, BinGroupsMultiSensorMemory binStructure, uint32 instantsProcessed, ResultArray out);


namespace AutocorrelationCUDA {

	/**
	* @brief Counts how many bin groups should be processed at the given instant.
	* @details Groups to process = max({k | instant % 2^k == 0}).
	* @tparam Integer Integer type of the instant.
	* @param x The instant.
	* @param bits Number of bits of the integer type representing the instant.
	* @return How many bin groups should be processed at the given instant.
	**/
	template<typename Integer>
	__device__ static Integer repeatTimes(Integer x, std::int8_t bits) {
		Integer pow2 = 1;
		for (std::uint8_t i = 0; i < bits; ++i) {
			if ((x & pow2) != 0) {
				return i + 1;
			}
			pow2 = pow2 << 1;
		}
		return bits;
	}
}


int main() {
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

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
	

	std::vector<uint8> dataDebug(SENSORS * INSTANTS_PER_PACKET);
	for (int i = 0; i < INSTANTS_PER_PACKET; ++i) {
		for (int j = 0; j < SENSORS; ++j) {
			dataDebug[i*SENSORS + j] = (i%7) +1;
		}
	}

	uint32 timesCalled; //counter
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
	
	out.download();	
	std::cout << "\Kernel called " << timesCalled << " times\n";
	for (int sensor = 0; sensor < 3; ++sensor) {
		std::cout << "\n\n\t======= SENSOR " << sensor << " =======\n";

		for (int lag = 0; lag < MAX_LAG; ++lag) {
			std::cout << "\n\t" << lag+1 << " --> " << out.get(sensor, lag);
		}
	}

	//write output to file
	//AutocorrelationCUDA::DataFile<std::uint_fast32_t>::write(out);
	
	cudaDeviceReset();
	return 0;
}




/**
* @brief Calculates the autocorrelation of the data in the packet. At the end of the processing it copies it to the host device.
* @details This CUDA function is optimized to work in the shortest time possible. 
		   To achieve this result all of the data in the binStructure and in the out array, is copied to the shared memory of the GPU, and copied back to the global memory at the end of the computation.
* @param packet New data to analize.
* @param binStructure To ensure a fast computation, the bigger is the lag, the more coalesced the data is. For example, if lag is 14, all of the instants in the packet array is treated singularly. Whereas, if lag is bigger (41), then 2 consequent intants in the packet array are summed togheter. binStructure is the object which encapsulates this concept.
* @param instantsProcessed How many instants have been already processed in previous calls to this function.
* @param out Output array, containing data computed in previous calls to this function.
**/
__global__ void autocorrelate(SensorsDataPacket packet, BinGroupsMultiSensorMemory binStructure, uint32 instantsProcessed, ResultArray out) {
	
	//precondition: blockDim.x = groupSize, blockDim.y = sensorsPerBlock

	uint16 absoluteY = threadIdx.y + blockIdx.x * blockDim.y;
	uint16 startingAbsoluteY = blockIdx.x * blockDim.y;
	uint8 relativeID = threadIdx.x + threadIdx.y * blockDim.x; //not more than 256 threads per block (basically 8 sensors)

	//put data in shared memory
	__shared__ uint16 binStruct[ELEMS_REQUIRED_FOR_BIN_STRUCTURE];
	uint16* data = binStruct;
	uint16* accumulatorsPos = &data[SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE];
	uint16* zeroDelays = &accumulatorsPos[SENSORS_PER_BLOCK * GROUPS_PER_SENSOR];
	
	__shared__ uint32 output[ELEMS_REQUIRED_FOR_OUTPUT];

	//copy data
	uint32* tmpArr1 = (uint32*)data;
	uint32* tmpArr2;
	for (int i = 0; i < COPY_REPETITIONS; ++i) {
		if(relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE < X32_BITS_PER_BLOCK_DATA){
			tmpArr1[relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE] = binStructure.rawGet(relativeID + blockIdx.x * X32_BITS_PER_BLOCK_DATA + i * SENSORS_PER_BLOCK * GROUP_SIZE);
		}
	}

	//copy output
	for (int i = 0; i < COPY_REPETITIONS * 2; ++i) {
		if (relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE < X32_BITS_PER_BLOCK_DATA * 2) {
			output[relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE] = out.rawGet(relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE + blockIdx.x * X32_BITS_PER_BLOCK_DATA * 2);
		}
	}


	//copy accumulatorsPos and zeroDelays
	tmpArr1 = (uint32*)accumulatorsPos;
	tmpArr2 = (uint32*)zeroDelays;
	if (relativeID < GROUPS_PER_SENSOR * SENSORS_PER_BLOCK / 2) {
		tmpArr1[relativeID] = binStructure.rawGetAccumulatorRelativePos(relativeID + blockIdx.x * X32_BITS_PER_BLOCK_ZD_ACC);
		tmpArr2[relativeID] = binStructure.rawGetZeroDelay(relativeID + blockIdx.x * X32_BITS_PER_BLOCK_ZD_ACC);
	}

	__syncthreads();



	//cycle over all of the new data, where i is the instant in time processed
	for (int i = 0; i < INSTANTS_PER_PACKET; ++i) {
		
		instantsProcessed++;
		//only one thread per sensor adds the new datum to the bin structure
		if (threadIdx.x < GROUPS_PER_SENSOR) {
			BinGroupsMultiSensorMemory::insertNew(threadIdx.y, threadIdx.x, packet.get(startingAbsoluteY + relativeID, i), data);
		}
		__syncthreads();
			
		//calculate autocorrelation for that instant
		//Decides how many group to calculate, based on how many instants have been already processed (i.e. 1 instant-->0; 2-->0,1; 3-->0; 4-->0,1,2; 5-->0; 6-->0,1; ...)
		uint32 repeatTimes = AutocorrelationCUDA::repeatTimes(instantsProcessed, 32);
		for (uint8 j = 0; j < repeatTimes && j < GROUPS_PER_SENSOR; ++j) {
			
			uint8 lastGroup = repeatTimes < GROUPS_PER_SENSOR ? repeatTimes - 1 : GROUPS_PER_SENSOR -1;

			ResultArray::get(threadIdx.y, (lastGroup-j) * GROUP_SIZE + threadIdx.x,  output) += BinGroupsMultiSensorMemory::getZeroDelay(threadIdx.y, lastGroup - j, data) * BinGroupsMultiSensorMemory::get(threadIdx.y, lastGroup - j, threadIdx.x, data);
			__syncthreads();

			//only one thread per sensor makes the shift
			if (relativeID < SENSORS_PER_BLOCK) {
				BinGroupsMultiSensorMemory::shift(relativeID, lastGroup - j, data);
			}
			__syncthreads();
		}

		
	}


	//copy data out
	tmpArr1 = (uint32*)data;
	for (int i = 0; i < COPY_REPETITIONS; ++i) {
		if (relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE < X32_BITS_PER_BLOCK_DATA) {
			binStructure.rawGet(relativeID + blockIdx.x * X32_BITS_PER_BLOCK_DATA + i * SENSORS_PER_BLOCK * GROUP_SIZE) = tmpArr1[relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE];
		}
	}

	//copy accumulatorsPos and zeroDelays out
	tmpArr1 = (uint32*)accumulatorsPos;
	tmpArr2 = (uint32*)zeroDelays;
	if (relativeID < GROUPS_PER_SENSOR * SENSORS_PER_BLOCK / 2) {
		binStructure.rawGetAccumulatorRelativePos(relativeID + blockIdx.x * X32_BITS_PER_BLOCK_ZD_ACC) = tmpArr1[relativeID];
		binStructure.rawGetZeroDelay(relativeID + blockIdx.x * X32_BITS_PER_BLOCK_ZD_ACC) = tmpArr2[relativeID];
	}

	//copy output
	for (int i = 0; i < COPY_REPETITIONS * 2; ++i) {
		if (relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE < X32_BITS_PER_BLOCK_DATA * 2) {
			out.rawGet(relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE + blockIdx.x * X32_BITS_PER_BLOCK_DATA * 2) = output[relativeID + i * SENSORS_PER_BLOCK * GROUP_SIZE];
		}
	}
}
