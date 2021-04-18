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

#define SENSORS_EXP2_DEFAULT 8
#define GROUPS_DEFAULT 10
#define INSTANTS_PER_PACKET_DEFAULT 80000
#define REPETITIONS_DEFAULT 4
#define GROUP_SIZE_EXP2 5


template <typename Contained, int SizeExp2>
__global__ void autocorrelate(AutocorrelationCUDA::SensorsDataPacket<Contained> packet, AutocorrelationCUDA::BinGroupsMultiSensorMemory<Contained, SizeExp2> binStructure, std::uint_fast32_t instantsProcessed, AutocorrelationCUDA::ResultArray<Contained> out);


namespace AutocorrelationCUDA {

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

}


std::vector<std::uint_fast32_t> askParameters() {
	std::vector<std::uint_fast32_t> result;
	std::uint_fast32_t tmp;

	std::cout << "\nInsert 0 for default parameter\n";
	std::cout << "\nnumber of sensors --> insert a number x, you'll have 2^x sensors: ";std::cin >> tmp;	result.emplace_back(tmp <= 0 ? SENSORS_EXP2_DEFAULT : tmp);
	std::cout << "\nnumber of bin groups: ";											std::cin >> tmp;	result.emplace_back(tmp <= 0 ? GROUPS_DEFAULT : tmp);
	std::cout << "\ninstants in a sigle packet: ";										std::cin >> tmp;	result.emplace_back(tmp <= 0 ? INSTANTS_PER_PACKET_DEFAULT : tmp);
	std::cout << "\nrepetitions: ";														std::cin >> tmp;	result.emplace_back(tmp <= 0 ? REPETITIONS_DEFAULT : tmp);

	return result;
}



int main() {
	
	//ask parameters to user
	std::vector<std::uint_fast16_t> params = askParameters();
	const std::uint_fast32_t sensorsExp2 = params[0];
	const std::uint_fast32_t groups = params[1];
	const std::uint_fast32_t instantsPerPacket = params[2];
	const std::uint_fast32_t repetitions = params[3];

	std::uint_fast32_t sensors = std::pow(2, sensorsExp2);

	//open file where data is stored
	//std::unique_ptr<AutocorrelationCUDA::CudaInput<int>> dataFile = std::make_unique<AutocorrelationCUDA::InputVector<int>>("", "test1");
	

	//create array where to put new data for the GPU
	AutocorrelationCUDA::SensorsDataPacket<std::uint_fast32_t> inputArray(sensorsExp2, instantsPerPacket);

	//array in GPU of the bin groups structure
	AutocorrelationCUDA::BinGroupsMultiSensorMemory<std::uint_fast32_t, GROUP_SIZE_EXP2> binStructure(sensors, groups);

	//output array to store results in GPU
	auto out = binStructure.generateResultArray();


	dim3 numberOfBlocks = sensors / binStructure.getSensorsPerBlock(); //number of blocks active on the GPU
	dim3 threadsPerBlock {binStructure.getSensorsPerBlock(), (unsigned int) std::pow(2, GROUP_SIZE_EXP2)}; 
	std::size_t sharedMemoryRequired = binStructure.getTotalSharedMemoryRequired();
	
	//timer
	AutocorrelationCUDA::Timer timer{[](std::vector<double> data){AutocorrelationCUDA::DataFile<double>::write(data, "out_timer.txt");},
									 [](){struct timeval tp;
									      gettimeofday(&tp, NULL);
									      return ((double)tp.tv_sec + (double)tp.tv_usec * 0.000001);}};
	

	std::vector<std::uint_fast32_t> dataDebug(sensors * instantsPerPacket, 14);
	std::uint_fast32_t timesCalled; //counter
	timer.start();
	for(timesCalled = 0; timesCalled < repetitions; ++timesCalled) {
		inputArray.setNewDataPacket(dataDebug); //store in GPU memory a new block of data to be processed
		//inputArray.setNewDataPacket(dataFile->read(sensors * instantsPerPacket)); //store in GPU memory a new block of data to be processed
		//cudaDeviceSynchronize();
		//timer.getInterval();
		//timer.start();
		autocorrelate <<< numberOfBlocks, threadsPerBlock, sharedMemoryRequired >>> (inputArray, binStructure, timesCalled * instantsPerPacket, out);
		cudaDeviceSynchronize();	
		timer.getInterval();
	}
	

	std::cout << timesCalled << "\n";
	for (int sensor = 0; sensor < out.getSensors(); ++sensor) {
		std::cout << "\n\n\t======= SENSOR " << sensor << " =======\n";

		for (int lag = 0; lag < out.getMaxLagv(); ++lag) {
			int curr = out.get(sensor, lag);
			int div = (timesCalled*instantsPerPacket) - lag;
			float print = (float) curr / div;
			std::cout << "\n\t" << lag+1 << " --> " << print;
		}
	}

	//write output to file
	//AutocorrelationCUDA::DataFile<std::uint_fast32_t>::write(out);

	cudaDeviceReset();
}



extern __shared__ std::uint_fast8_t sharedMemory[];

template <typename Contained, int SizeExp2>
__global__ void autocorrelate(AutocorrelationCUDA::SensorsDataPacket<Contained> packet, AutocorrelationCUDA::BinGroupsMultiSensorMemory<Contained, SizeExp2> binStructure, std::uint_fast32_t instantsProcessed, AutocorrelationCUDA::ResultArray<Contained> out) {
	
	//precondition: blockDim.x = groupSize, blockDim.y = sensorsPerBlock

	std::uint_fast16_t blockDimTot = blockDim.x * blockDim.y;
	std::uint_fast16_t absoluteY = threadIdx.y + blockIdx.x * blockDim.y;
	std::uint_fast16_t absoluteID = threadIdx.x + absoluteY * blockDim.x;
	std::uint_fast16_t relativeID = threadIdx.x + threadIdx.y * blockDim.x;

	//put data in shared memory
	//TODO probably it is better to load only the most used groups (0, 1, 2, 3), to increase occupancy
	Contained* data = (Contained*)sharedMemory;
	std::uint_fast8_t* accumulatorsPos = (std::uint_fast8_t*) &data[blockDim.x * blockDim.y * binStructure.groupsNum()];
	Contained* zeroDelays = (Contained*) &accumulatorsPos[blockDim.y * binStructure.groupsNum()];
	std::uint_fast32_t* info = (std::uint_fast32_t*) & zeroDelays[blockDim.y * binStructure.groupsNum()];
	Contained* output = (Contained*) &info[7];

	//copy data
	for (int group = 0; group < binStructure.groupsNum(); ++group) {
		data[relativeID] = binStructure.get(absoluteY, group, threadIdx.x);
	}

	//copy accumulatorsPos (groupsNum <= groupSize)
	if (relativeID < binStructure.groupsPerBlock()) {
		accumulatorsPos[relativeID] = binStructure.getAccumulatorRelativePos(absoluteY, threadIdx.x);
	}

	//copy zeroDelays
	if (relativeID < binStructure.groupsPerBlock()) {
		zeroDelays[relativeID] = binStructure.getZeroDelay(absoluteY, threadIdx.x);
	}

	//copy info
	if (relativeID < 7) {
		info[relativeID] = binStructure.getInfo(relativeID);
	}



	//we only have e.g. 31 values per group + 1 accumulator. This means that in the output there will be "holes" at positions multiple of e.g. 32
	if(threadIdx.x != blockDim.x){
		Contained instantsNum = packet.instantsNum();
		//cycle over all of the new data, where i is the instant in time processed
		for (int i = 0; i < instantsNum; ++i) {
			instantsProcessed++;

			//only one thread per sensor adds the new datum to the bin structure
			if (threadIdx.x == 0) {
				binStructure.insertNew(threadIdx.y, packet.get(absoluteY, i), data, zeroDelays, accumulatorsPos, info);
			}

			//calculate autocorrelation for that instant
			//Decides how many group to calculate, based on how many instants have been already processed (i.e. 1 instant-->0; 2-->0,1; 3-->0; 4-->0,1,2; 5-->0; 6-->0,1; ...)
			std::uint_fast32_t repeatTimes = AutocorrelationCUDA::repeatTimes(instantsProcessed, 32);
			for (std::uint_fast8_t j = 0; j < repeatTimes; ++j) {

				output[relativeID + blockDimTot * j] += binStructure.getZeroDelay(blockIdx.y, j) * binStructure.get(blockIdx.y, j, threadIdx.x);
			
				//only one thread per sensor makes the shift
				if (threadIdx.x == 0) {
					binStructure.shift(blockIdx.y, j, data, zeroDelays, accumulatorsPos, info);
				}
			}
		}
	}

	//copy output to total output
	for (int i = 0; i < binStructure.groupsNum(); ++i) {
		out.addTo(absoluteY, threadIdx.x + blockDim.x * i, output[relativeID + blockDim.x * i]);
	}
}

