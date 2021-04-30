#ifndef AUTOCORRELATIONCUDA_BINGROUPSMULTISENSORMEMORY
#define AUTOCORRELATIONCUDA_BINGROUPSMULTISENSORMEMORY


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <cmath>
#include <iostream>

#include "Definitions.h"
#include "ResultArray.h"



namespace AutocorrelationCUDA {


/*
	
	FIXME restore old layout (block layout)


		 	+-----------+-----------+-----------+-----------+
			|zero delay	|zero delay	|	...		|zero delay	|
			|s.0 g.0	|s.1 g.0	|			|s.k g.0	|
			+-----------+-----------+-----------+-----------+
			|zero delay	|zero delay	|	...		|zero delay	|
			|s.0 g.1	|s.1 g.1	|			|s.k g.1	|
			+-----------+-----------+-----------+-----------+
			|	...		|	...		|	...		|	...		|
			|			|			|			|			|
			+-----------+-----------+-----------+-----------+
			|zero delay	|zero delay	|	...		|zero delay	|
			|s.0 g.h	|s.1 g.h	|			|s.k g.h	|
		 	+-----------+-----------+-----------+-----------+







		 	+-----------+-----------+-----------+-----------+
			|accumulator|accumulator|	...		|accumulator|
			|s.0 g.0	|s.1 g.0	|			|s.k g.0	|
			+-----------+-----------+-----------+-----------+
			|accumulator|accumulator|	...		|accumulator|
			|s.0 g.1	|s.1 g.1	|			|s.k g.1	|
			+-----------+-----------+-----------+-----------+
			|	...		|	...		|	...		|	...		|
			|			|			|			|			|
			+-----------+-----------+-----------+-----------+
			|accumulator|accumulator|	...		|accumulator|
			|s.0 g.h	|s.1 g.h	|			|s.k g.h	|
			+-----------+-----------+-----------+-----------+








	next array structure eliminates bank conflicts in CUDA shared memory
																																											
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	s.0 g.0	|	s.0 g.0	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	pos x	|	pos x+1	|			|	s.0 g.0	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	s.0 g.1	|	s.0 g.1	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	pos x	|	pos x+1	|			|	s.0 g.1	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	s.0 g.2	|	s.0 g.2	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	pos x	|	pos x+1	|			|	s.0 g.2	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	...		|	...		|	...		|	...		|	...		||	...		|	...		|	...		|	...		|	...		||	...		||	...		|	...		|	...		|	...		|	...		|
		|			|			|			|			|			||			|			|			|			|			||			||			|			|			|			|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	s.0 g.h	|	s.0 g.h	|	...		|	accum.	|	...		||	s.1 g.h	|	s.1 g.h	|	...		|	accum.	|	...		||	...		||	s.k g.h	|	s.k g.h	|	...		|	accum.	|	...		|
		|	pos x	|	pos x+1	|			|	s.0 g.h	|			||	pos x	|	pos x+1	|			|	s.1 g.h	|			||			||	pos x	|	pos x+1	|			|	s.k g.h	|			|
		+===========+===========+===========+===========+===========++==========+===========+===========+===========+===========++==========++==========+===========+===========+===========+===========+
		|	s.0 g.0	|	s.0 g.0	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	pos x	|	pos x+1	|			|	s.0 g.0	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	s.0 g.1	|	s.0 g.1	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	pos x	|	pos x+1	|			|	s.0 g.1	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	s.0 g.2	|	s.0 g.2	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	pos x	|	pos x+1	|			|	s.0 g.2	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	...		|	...		|	...		|	...		|	...		||	...		|	...		|	...		|	...		|	...		||	...		||	...		|	...		|	...		|	...		|	...		|
		|			|			|			|			|			||			|			|			|			|			||			||			|			|			|			|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	s.0 g.h	|	s.0 g.h	|	...		|	accum.	|	...		||	s.1 g.h	|	s.1 g.h	|	...		|	accum.	|	...		||	...		||	s.k g.h	|	s.k g.h	|	...		|	accum.	|	...		|
		|	pos x	|	pos x+1	|			|	s.0 g.h	|			||	pos x	|	pos x+1	|			|	s.1 g.h	|			||			||	pos x	|	pos x+1	|			|	s.k g.h	|			|
		+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+









*/


class BinGroupsMultiSensorMemory final {

	public:

	__host__ BinGroupsMultiSensorMemory() {
		
		std::cout << "\ninitializing BinGroupMultiSensor...\n";

		//create matrix for data on GPU and fill it with 0
		std::cout << "\nallocating data area on GPU\n";
		cudaMalloc(&data, GROUP_SIZE * GROUPS_PER_SENSOR * SENSORS * sizeof(uint16));
		cudaMemset(data, 0, GROUP_SIZE * GROUPS_PER_SENSOR * SENSORS * sizeof(uint16));

		//create matrix for zero delay data on GPU and fill it with 0
		std::cout << "\allocating zero delay area on GPU\n";
		cudaMalloc(&zeroDelays, GROUPS_PER_SENSOR * SENSORS * sizeof(uint16));
		cudaMemset(zeroDelays, 0, GROUPS_PER_SENSOR * SENSORS * sizeof(uint16));

		//create matrix for accumulator positions for each group on GPU and fill it with 0
		std::cout << "\nallocating accumulators on GPU\n";
		cudaMalloc(&accumulatorsPos, GROUPS_PER_SENSOR * SENSORS * sizeof(uint16));
		cudaMemset(accumulatorsPos, 0, GROUPS_PER_SENSOR * SENSORS * sizeof(uint16));

		std::cout << "\nBinGroupMultiSensor done!\n";
	}




	__host__ ResultArray generateResultArray() {
		return ResultArray();
	}





	
	__device__ static uint16& getAccumulatorRelativePos(uint16 sensor, uint16 group, uint16* arr) {
		return arr[ACC_POS_START + sensor + group * SENSORS_PER_BLOCK];
	}


	__device__ static uint16& getZeroDelay(uint16 sensor, uint16 group, uint16* arr) {
		return arr[ZERO_DELAY_START + sensor + group * SENSORS_PER_BLOCK];
	}


	__device__ static uint16& get(uint16 sensor, uint16 group, uint16 pos, uint16* arr) {
		return arr[((getAccumulatorRelativePos(sensor, group, arr)+1+pos) & (GROUP_SIZE-1)) + sensor * GROUP_SIZE + group * SENSORS_PER_BLOCK * GROUP_SIZE];
	}


	__device__ static uint16& getAccumulator(uint16 sensor, uint16 group, uint16* arr) {
		return arr[getAccumulatorRelativePos(sensor, group, arr) + sensor * GROUP_SIZE + group * SENSORS_PER_BLOCK * GROUP_SIZE];
	}


	__device__ static void insertNew(uint16 sensor, uint16 datum, uint16* arr) {
		getAccumulator(sensor, 0, arr) = datum;
		getZeroDelay(sensor, 0, arr) = datum;
	}


	__device__ static void shift(uint16 sensor, uint16 group, uint16* arr) {
		getAccumulatorRelativePos(sensor, group, arr) = (getAccumulatorRelativePos(sensor, group, arr)-1)&(GROUP_SIZE-1); //decrement accumulator pos

		if (group < GROUP_SIZE - 1) {
			getAccumulator(sensor, group+1, arr) += getAccumulator(sensor, group, arr);
			getZeroDelay(sensor, group+1, arr) += getZeroDelay(sensor, group, arr);
		}

		getAccumulator(sensor, group, arr) = 0;
		getZeroDelay(sensor, group, arr) = 0;
	}





	/*__device__ uint16& getAccumulatorRelativePos(uint16 sensor, uint16 group) {
		return accumulatorsPos[sensor + group * SENSORS];
	}*/

	__device__ uint32& rawGetAccumulatorRelativePos(uint16 i) {
		uint32* tmp = (uint32*)accumulatorsPos;
		return tmp[i];
	}


	/*__device__ uint16& getZeroDelay(uint16 sensor, uint16 group) {
		return zeroDelays[sensor + group * SENSORS];
	}*/

	__device__ uint32& rawGetZeroDelay(uint16 i) {
		uint32* tmp = (uint32*)zeroDelays;
		return tmp[i];
	}

	/*__device__ uint16& get(uint16 sensor, uint16 group, uint16 pos) {
		return data[(getAccumulatorRelativePos(sensor, group) + 1 + pos) & (GROUP_SIZE - 1) + sensor + group * SENSORS * GROUP_SIZE];
	}*/

	__device__ uint32& rawGet(uint16 i) {
		uint32* tmp = (uint32*)data;
		return tmp[i];
	}



	//arrays in GPU global memory
	uint16* data;
	uint16* zeroDelays;
	uint16* accumulatorsPos;
};

}

#endif
