#ifndef AUTOCORRELATIONCUDA_RESULTARRAY
#define AUTOCORRELATIONCUDA_RESULTARRAY


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>

#include "Definitions.h"


namespace AutocorrelationCUDA {


/*


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
                                            		 
*/


class ResultArray final {
	
	public:

	__host__ ResultArray() {

		std::cout << "\ninitializing ResultArray...\n";

		 cudaMalloc(&data, SENSORS * MAX_LAG * sizeof(uint32));
		 cudaMemset(data, 0, SENSORS * MAX_LAG * sizeof(uint32));

		 hostData = (uint32*)malloc(SENSORS * MAX_LAG * sizeof(uint32));

		 std::cout << "\nResultArray done!\n";
	}



	__device__ static uint32& get(uint16 sensor, uint16 lag, uint32* arr) {
		return arr[sensor * MAX_LAG + lag];
	}


	__device__ void addTo(uint16 sensor, uint16 lag, uint16 datum) {
		data[sensor * MAX_LAG + lag] += datum;
	}


	__host__ uint32 get(uint16 sensor, uint16 lag) {
		return hostData[sensor * MAX_LAG + lag];
	}


	__host__ void download() {		
		cudaMemcpy(hostData, data, MAX_LAG * SENSORS * sizeof(uint32), cudaMemcpyDeviceToHost);
	}




	private:

	uint32* data;

	uint32* hostData;

};
}


#endif
