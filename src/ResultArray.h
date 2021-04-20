#ifndef AUTOCORRELATIONCUDA_RESULTARRAY
#define AUTOCORRELATIONCUDA_RESULTARRAY


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>
#include "BinGroupsMultiSensorMemory.h"


namespace AutocorrelationCUDA {

#define MAX_LAG (GROUPS_PER_SENSOR * GROUP_SIZE)




/*

		 sens.0	 sens.1	 sens.2	 sens.3	 sens.4	...
		+-------+-------+-------+-------+-------+----------
		|	lag	|	lag	|	lag	|	lag	|	lag	|	lag	
		|	1	|	1	|	1	|	1	|	1	|	...	
		+-------+-------+-------+-------+-------+----------
		|	lag	|	lag	|	lag	|	lag	|	lag	|	lag
		|	2	|	2	|	2	|	2	|	2	|	...
		+-------+-------+-------+-------+-------+----------
		|	lag	|	lag	|	lag	|	lag	|	lag	|	lag
		|	3	|	3	|	3	|	3	|	3	|	...
		+-------+-------+-------+-------+-------+----------
		|	lag	|	lag	|	lag	|	lag	|	lag	|	lag
		|	4	|	4	|	4	|	4	|	4	|	...
		+-------+-------+-------+-------+-------+----------
		|	...	|	...	|	...	|	...	|	...	|	...	
		|		|		|		|		|		|		

*/


class ResultArray final {
	
	public:

	__host__ ResultArray() {

		std::cout << "\ninitializing ResultArray...\n";

		 cudaMalloc(&data, SENSORS * MAX_LAG * sizeof(uint32));
		 cudaMemset(data, 0, SENSORS * MAX_LAG * sizeof(uint32));

		 std::cout << "\nResultArray done!\n";
	}



	__device__ static void addTo(uint16 sensor, uint8 lag, uint32 datum, uint32* arr) {
		arr[sensor + lag * MAX_LAG] += datum;
	}


	__device__ void addTo(uint16 sensor, uint8 lag, uint32 datum) {
		data[sensor + lag * MAX_LAG] += datum;
	}


	__host__ uint32 get(uint16 sensor, uint8 lag) {
		return toVector()[sensor * MAX_LAG + lag];
	}


	__host__ std::vector<uint32> toVector() {		
		std::vector<uint32> result(MAX_LAG * SENSORS);
		cudaMemcpy(result.data(), data, MAX_LAG * SENSORS * sizeof(uint32), cudaMemcpyDeviceToHost);
		return result;
	}




	private:

	uint32* data;

};
}


#endif
