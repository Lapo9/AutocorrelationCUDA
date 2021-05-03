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


/**	

	next array structure eliminates bank conflicts in CUDA shared memory
		
		assumed 8 sensors per block, 10 bin groups, group size 32, initial state (accumulators on first position)
																																											
		+-----------+-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+-----------+
		|  accum.   |  s.0 g.0  |  s.0 g.0  |  ...      |  s.0 g.0  ||  accum.   |  s.1 g.0  |  s.1 g.0  |  ...      |  s.1 g.0  ||  ...      ||  accum.   |  s.7 g.0  |  s.7 g.0  |  ...      |  s.7 g.0  |
		|  s.0 g.0  |  pos 1    |  pos 2    |           |  pos 32   ||  s.1 g.0  |  pos 1    |  pos 2    |           |  pos 32   ||  ...      ||  s.7 g.0  |  pos 1    |  pos 2    |           |  pos 32   |
		+-----------+-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+-----------+
		|  accum.   |  s.0 g.1  |  s.0 g.1  |  ...      |  s.0 g.1  ||  accum.   |  s.1 g.1  |  s.1 g.1  |  ...      |  s.1 g.1  ||  ...      ||  accum.   |  s.7 g.1  |  s.7 g.1  |  ...      |  s.7 g.1  |
		|  s.0 g.1  |  pos 1    |  pos 2    |           |  pos 32   ||  s.1 g.1  |  pos 1    |  pos 2    |           |  pos 32   ||  ...      ||  s.7 g.1  |  pos 1    |  pos 2    |           |  pos 32   |
		+-----------+-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+-----------+
		|  ...      |  ...      |  ...      |  ...      |  ...      ||  ...      |  ...      |  ...      |  ...      |  ...      ||  ...      ||  ...      |  ...      |  ...      |  ...      |  ...      |
		|           |           |           |           |           ||           |           |           |           |           ||  ...      ||           |           |           |           |           |
		+-----------+-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+-----------+
		|  accum.   |  s.0 g.9  |  s.0 g.9  |  ...      |  s.0 g.9  ||  accum.   |  s.1 g.9  |  s.1 g.9  |  ...      |  s.1 g.9  ||  ...      ||  accum.   |  s.7 g.9  |  s.7 g.9  |  ...      |  s.7 g.9  |
		|  s.0 g.9  |  pos 1    |  pos 2    |           |  pos 32   ||  s.1 g.9  |  pos 1    |  pos 2    |           |  pos 32   ||  ...      ||  s.7 g.9  |  pos 1    |  pos 2    |           |  pos 32   |
		+===========+===========+===========+===========+===========++===========+===========+===========+===========+===========++===========++===========+===========+===========+===========+===========+
		|  accum.   |  s.8 g.0  |  s.8 g.0  |  ...      |  s.8 g.0  ||  accum.   |  s.9 g.0  |  s.9 g.0  |  ...      |  s.9 g.0  ||  ...      ||  accum.   |  s.15 g.0 |  s.15 g.0 |  ...      |  s.15 g.0 |
		|  s.8 g.0  |  pos 1    |  pos 2    |           |  pos 32   ||  s.9 g.0  |  pos 1    |  pos 2    |           |  pos 32   ||  ...      ||  s.15 g.0 |  pos 1    |  pos 2    |           |  pos 32   |
		+-----------+-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+-----------+
		|  accum.   |  s.8 g.1  |  s.8 g.1  |  ...      |  s.8 g.1  ||  accum.   |  s.9 g.1  |  s.9 g.1  |  ...      |  s.9 g.1  ||  ...      ||  accum.   |  s.15 g.1 |  s.15 g.1 |  ...      |  s.15 g.1 |
		|  s.8 g.1  |  pos 1    |  pos 2    |           |  pos 32   ||  s.9 g.1  |  pos 1    |  pos 2    |           |  pos 32   ||  ...      ||  s.15 g.1 |  pos 1    |  pos 2    |           |  pos 32   |
		+-----------+-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+-----------+
		|  ...      |  ...      |  ...      |  ...      |  ...      ||  ...      |  ...      |  ...      |  ...      |  ...      ||  ...      ||  ...      |  ...      |  ...      |  ...      |  ...      |
		|           |           |           |           |           ||           |           |           |           |           ||  ...      ||           |           |           |           |           |
		+-----------+-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+-----------+
		|  accum.   |  s.8 g.9  |  s.8 g.9  |  ...      |  s.8 g.9  ||  accum.   |  s.9 g.9  |  s.9 g.9  |  ...      |  s.9 g.9  ||  ...      ||  accum.   |  s.15 g.9 |  s.15 g.9 |  ...      |  s.15 g.9 |
		|  s.8 g.9  |  pos 1    |  pos 2    |           |  pos 32   ||  s.9 g.9  |  pos 1    |  pos 2    |           |  pos 32   ||  ...      ||  s.15 g.9 |  pos 1    |  pos 2    |           |  pos 32   |
		+===========+===========+===========+===========+===========++===========+===========+===========+===========+===========++===========++===========+===========+===========+===========+===========+
		|  ...      |  ...      |  ...      |  ...      |  ...      ||  ...      |  ...      |  ...      |  ...      |  ...      ||  ...      ||  ...      |  ...      |  ...      |  ...      |  ...      |
		|  ...      |  ...      |  ...      |  ...      |  ...      ||  ...      |  ...      |  ...      |  ...      |  ...      ||  ...      ||  ...      |  ...      |  ...      |  ...      |  ...      |        





		assumed 8 sensors per block, 10 bin groups

		+-----------+-----------+-----------+-----------+
		|accumulator|accumulator|   ...     |accumulator|
		|s.0 g.0    |s.1 g.0    |           |s.7 g.0    |
		+-----------+-----------+-----------+-----------+
		|accumulator|accumulator|   ...     |accumulator|
		|s.0 g.1    |s.1 g.1    |           |s.7 g.1    |
		+-----------+-----------+-----------+-----------+
		|   ...     |   ...     |   ...     |   ...     |
		|           |           |           |           |
		+-----------+-----------+-----------+-----------+
		|accumulator|accumulator|   ...     |accumulator|
		|s.0 g.9    |s.1 g.9    |           |s.7 g.9    |
		+===========+===========+===========+===========+
		|accumulator|accumulator|   ...     |accumulator|
		|s.8 g.0    |s.9 g.0    |           |s.15 g.0   |
		+-----------+-----------+-----------+-----------+
		|accumulator|accumulator|   ...     |accumulator|
		|s.8 g.1    |s.9 g.1    |           |s.15 g.1   |
		+-----------+-----------+-----------+-----------+
		|   ...     |   ...     |   ...     |   ...     |
		|           |           |           |           |
		+-----------+-----------+-----------+-----------+
		|accumulator|accumulator|   ...     |accumulator|
		|s.8 g.9    |s.9 g.9    |           |s.15 g.9   |
		+===========+===========+===========+===========+
		|   ...     |   ...     |   ...     |   ...     |
		|   ...     |   ...     |   ...     |   ...     |





		assumed 8 sensors per block, 10 bin groups

		+-----------+-----------+-----------+-----------+
		|zero delay |zero delay |   ...     |zero delay |
		|s.0 g.0    |s.1 g.0    |           |s.7 g.0    |
		+-----------+-----------+-----------+-----------+
		|zero delay |zero delay |   ...     |zero delay |
		|s.0 g.1    |s.1 g.1    |           |s.7 g.1    |
		+-----------+-----------+-----------+-----------+
		|   ...     |   ...     |   ...     |   ...     |
		|           |           |           |           |
		+-----------+-----------+-----------+-----------+
		|zero delay |zero delay |   ...     |zero delay |
		|s.0 g.9    |s.1 g.9    |           |s.7 g.9    |
		+===========+===========+===========+===========+
		|zero delay |zero delay |   ...     |zero delay |
		|s.8 g.0    |s.9 g.0    |           |s.15 g.0   |
		+-----------+-----------+-----------+-----------+
		|zero delay |zero delay |   ...     |zero delay |
		|s.8 g.1    |s.9 g.1    |           |s.15 g.1   |
		+-----------+-----------+-----------+-----------+
		|   ...     |    ...    |   ...     |   ...     |
		|           |           |           |           |
		+-----------+-----------+-----------+-----------+
		|zero delay |zero delay |   ...     |zero delay |
		|s.8 g.9    |s.9 g.9    |           |s.15 g.9   |
		+===========+===========+===========+===========+
		|   ...     |   ...     |   ...     |   ...     |
		|   ...     |   ...     |   ...     |   ...     |


**/
class BinGroupsMultiSensorMemory final {

	public:

	/**
	 * @brief Creates the structure described in the class description in the GPU global memory, and initializes it wizth zeroes.
	 */
	__host__ BinGroupsMultiSensorMemory() {
		
		std::cout << "\ninitializing BinGroupMultiSensor...\n";

		//create matrix for data on GPU and fill it with 0
		std::cout << "\nallocating data area on GPU\n";
		cudaMalloc(&data, GROUP_SIZE * GROUPS_PER_SENSOR * SENSORS * sizeof(uint16));
		cudaMemset(data, 6, GROUP_SIZE * GROUPS_PER_SENSOR * SENSORS * sizeof(uint16));

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



	/**
	 * @brief Generates a ResultArray object with the right dimensions for this BinGroupsMultiSensorMemory.
	 * @return A ResultArray object with the right dimensions for this BinGroupsMultiSensorMemory.
	 */
	__host__ ResultArray generateResultArray() {
		return ResultArray();
	}





	/**
	 * @brief Returns a reference to the accumulator relative position in the specified group of the specifeid sensor.
	 * @details The number of sensors per block is determined by the relative constexpr in Definitions.h.
	 * @param sensor Sensor. 
	 * @param group Bin group.
	 * @param arr Pointer to the first position of the array containing the portion of the BinGroupsMultiSensorMemory allocated in the GPU shared memory.
	 * @return A reference to the accumulator relative position in the specified group of the specifeid sensor.
	 * @pre sensor < SENSORS_PER_BLOCK, group < GROUPS_PER_SENSOR
	 */
	__device__ static uint16& getAccumulatorRelativePos(uint16 sensor, uint16 group, uint16* arr) {
		return arr[ACC_POS_START + sensor + group * SENSORS_PER_BLOCK];
	}


	/**
	* @brief Returns a reference to the zero delay value of the specified group of the specifeid sensor.
	* @details The number of sensors per block is determined by the relative constexpr in Definitions.h.
	* @param sensor Sensor.
	* @param group Bin group.
	* @param arr Pointer to the first position of the array containing the portion of the BinGroupsMultiSensorMemory allocated in the GPU shared memory.
	* @return A reference to the zero delay value of the specified group of the specifeid sensor.
	* @pre sensor < SENSORS_PER_BLOCK, group < GROUPS_PER_SENSOR
	*/
	__device__ static uint16& getZeroDelay(uint16 sensor, uint16 group, uint16* arr) {
		return arr[ZERO_DELAY_START + sensor + group * SENSORS_PER_BLOCK];
	}


	/**
	* @brief Returns a reference to value contained in the specified position of the specified group of the specifeid sensor.
	* @details The number of sensors per block and the group size are determined by the relative constexpr(es) in Definitions.h.
	* @param sensor Sensor.
	* @param group Bin group.
	* @param pos Position.
	* @param arr Pointer to the first position of the array containing the portion of the BinGroupsMultiSensorMemory allocated in the GPU shared memory.
	* @return A reference to value contained in the specified position of the specified group of the specifeid sensor.
	* @pre sensor < SENSORS_PER_BLOCK, group < GROUPS_PER_SENSOR, pos < GROUP_SIZE
	*/
	__device__ static uint16& get(uint16 sensor, uint16 group, uint16 pos, uint16* arr) {
		return arr[((getAccumulatorRelativePos(sensor, group, arr)+pos) & (GROUP_SIZE-1)) + sensor * GROUP_SIZE + group * SENSORS_PER_BLOCK * GROUP_SIZE];
	}


	/**
	* @brief Returns a reference to value contained in the accumulator of the specified group of the specifeid sensor.
	* @details The number of sensors per block and the group size are determined by the relative constexpr(es) in Definitions.h.
	* @param sensor Sensor.
	* @param group Bin group.
	* @param arr Pointer to the first position of the array containing the portion of the BinGroupsMultiSensorMemory allocated in the GPU shared memory.
	* @return A reference to value contained in the accumulator of the specified group of the specifeid sensor.
	* @pre sensor < SENSORS_PER_BLOCK, group < GROUPS_PER_SENSOR
	*/
	__device__ static uint16& getAccumulator(uint16 sensor, uint16 group, uint16* arr) {
		return arr[getAccumulatorRelativePos(sensor, group, arr) + sensor * GROUP_SIZE + group * SENSORS_PER_BLOCK * GROUP_SIZE];
	}



	/**
	 * @brief Insert a new value into the portion of this BinGroupsMultiSensorMemory conatined in the GPU shared memory a new value.
	 * @details The new value is inserted into the zero delay and the accumulator of the first group of the specified sensor.
	 * @param sensor Sensor to insert the new value into.
	 * @param datum Value to insert.
	 * @param arr Pointer to the first position of the array containing the portion of the BinGroupsMultiSensorMemory allocated in the GPU shared memory.
	 * @pre sensor < SENSORS_PER_BLOCK
	 */
	__device__ static void insertNew(uint16 sensor, uint16 datum, uint16* arr) {
		getAccumulator(sensor, 0, arr) = datum;
		getZeroDelay(sensor, 0, arr) = datum;
	}


	/**
	 * @brief Shifts the memory of the specified group of the specified sensor.
	 * @details The value that "exits" from the group is added to the next group accumulator.
	 *			The zero delay register of th group is added to the next group zero delay register.
	 *			Accumulator and zero delay register of the group are cleared.
	 * @param sensor Sensor.
	 * @param group	Bin group to shift.
	 * @param arr Pointer to the first position of the array containing the portion of the BinGroupsMultiSensorMemory allocated in the GPU shared memory.
	 * @pre sensor < SENSORS_PER_BLOCK, group < GROUPS_PER_SENSOR
	 */
	__device__ static void shift(uint16 sensor, uint16 group, uint16* arr) {
		getAccumulatorRelativePos(sensor, group, arr) = (getAccumulatorRelativePos(sensor, group, arr)-1)&(GROUP_SIZE-1); //decrement accumulator pos

		if (group < GROUPS_PER_SENSOR - 1) {
			getAccumulator(sensor, group+1, arr) += getAccumulator(sensor, group, arr);
			getZeroDelay(sensor, group+1, arr) += getZeroDelay(sensor, group, arr);
		}

		getAccumulator(sensor, group, arr) = 0;
		getZeroDelay(sensor, group, arr) = 0;
	}




	/**
	 * @brief Returns a reference to the i-th 32-bits integer of the data array.
	 * @details This method is provided to ensure the fastest access possible to the GPU global memory. 
	 *			The logic of the memory (the position of each datum based on the sensor and bin group) here is not taken into consideration, and the results are provided as is (the i-th element is actually the i-th 32-bits integer in the data array).
	 * @param i i-th 32-bit integer in data array.
	 * @return A reference to the i-th 32-bits integer of data.
	 * @pre i < ceil(data.length / 2)
	 */
	__device__ uint32& rawGetAccumulatorRelativePos(uint32 i) {
		uint32* tmp = (uint32*)accumulatorsPos;
		return tmp[i];
	}


	/**
	* @brief Returns a reference to the i-th 32-bits integer of the zero delay array.
	* @details This method is provided to ensure the fastest access possible to the GPU global memory.
	*			The logic of the memory (the position of each datum based on the sensor and bin group) here is not taken into consideration, and the results are provided as is (the i-th element is actually the i-th 32-bits integer in the zero delay array).
	* @param i i-th 32-bit integer in data array.
	* @return A reference to the i-th 32-bits integer of data.
	* @pre i < ceil(zeroDelays.length / 2)
	*/
	__device__ uint32& rawGetZeroDelay(uint32 i) {
		uint32* tmp = (uint32*)zeroDelays;
		return tmp[i];
	}


	/**
	* @brief Returns a reference to the i-th 32-bits integer of the accumulators position array.
	* @details This method is provided to ensure the fastest access possible to the GPU global memory.
	*			The logic of the memory (the position of each datum based on the sensor and bin group) here is not taken into consideration, and the results are provided as is (the i-th element is actually the i-th 32-bits integer in the accumulators position array).
	* @param i i-th 32-bit integer in data array.
	* @return A reference to the i-th 32-bits integer of data.
	* @pre i < ceil(accumulatorsPos.length / 2)
	*/
	__device__ uint32& rawGet(uint32 i) {
		uint32* tmp = (uint32*)data;
		return tmp[i];
	}



	private:

	//arrays in GPU global memory
	uint16* data;
	uint16* zeroDelays;
	uint16* accumulatorsPos;
};

}

#endif
