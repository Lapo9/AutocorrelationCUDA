#ifndef AUTOCORRELATIONCUDA_RESULTARRAY
#define AUTOCORRELATIONCUDA_RESULTARRAY


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>

#include "Definitions.h"


namespace AutocorrelationCUDA {


/**
	@brief Class responsible for storing the data processed in the GPU and being able to retrieve it on the host.

	@details
	The memory layout of the array is optimized for high read/write throughput:

            +-------+-------+-------+-------
    sensor  |  lag  |  lag  |  lag  |  ...    
        0   |  1    |  2    |  3    |        
            +-------+-------+-------+-------
    sensor  |  lag  |  lag  |  lag  |  ...    
        1   |  1    |  2    |  3    |        
            +-------+-------+-------+-------
    sensor  |  lag  |  lag  |  lag  |  ...    
        2   |  1    |  2    |  3    |        
            +-------+-------+-------+-------
    sensor  |  lag  |  lag  |  lag  |  ...    
        3   |  1    |  2    |  3    |        
            +-------+-------+-------+-------
    ...     |  ...  |  ...  |  ...  |  ...    
            |       |       |       |        
                                            		 
**/
class ResultArray final {
	
	public:

	/**
	* @brief Creates an array on the CUDA GPU of size SENSORS * MAX_LAG and fills it with 0. Then creates the same array on the host memory.
	**/
	__host__ ResultArray() {

		std::cout << "\ninitializing ResultArray...\n";

		 cudaMalloc(&data, SENSORS * MAX_LAG * sizeof(uint32));
		 cudaMemset(data, 0, SENSORS * MAX_LAG * sizeof(uint32));

		 hostData = (uint32*)malloc(SENSORS * MAX_LAG * sizeof(uint32));

		 std::cout << "\nResultArray done!\n";
	}



	/**
	* @brief Returns a reference to the element of the specified sensor at the specified lag.
	* @param sensor Sensor.
	* @param lag Lag.
	* @param arr Array stored on the GPU where to search for the specified element. Usually this array is stored on shared memory, so it doesn't correspond to the data array of this class.
	* @return a reference to the element of the specified sensor at the specified lag.
	* @pre sensor * lag < arr.length()
	**/
	__device__ static uint32& get(uint16 sensor, uint16 lag, uint32* arr) {
		return arr[sensor * MAX_LAG + lag];
	}


	/**
	 * @brief Returns a reference to the i-th 32-bits integer of the data array.
	 * @details This method is provided to ensure the fastest access possible to the GPU global memory.
	 *			The logic of the memory (the position of each datum based on the sensor and bin group) here is not taken into consideration, and the results are provided as is (the i-th element is actually the i-th 32-bits integer in the data array).
	 * @param i i-th 32-bit integer in data array.
	 * @return A reference to the i-th 32-bits integer of data.
	 * @pre i < data.length
	 **/
	__device__ uint32& rawGet(uint32 i) {
		return data[i];
	}


	/**
	* @brief Returns a reference to the element of the specified sensor at the specified lag of the data array of this object.
	* @param sensor Sensor.
	* @param lag Lag.
	* @return A reference to the element of the specified sensor at the specified lag of the data array of this object.
	* @pre sensor * lag < SENSORS * MAX_LAG
	**/
	__host__ uint32 get(uint16 sensor, uint16 lag) {
		return hostData[sensor * MAX_LAG + lag];
	}


	/**
	* Copies all of the data from the array on the GPU to the array on the host device.
	**/
	__host__ void download() {		
		cudaMemcpy(hostData, data, MAX_LAG * SENSORS * sizeof(uint32), cudaMemcpyDeviceToHost);
	}




	private:

	uint32* data;

	uint32* hostData;

};
}


#endif
