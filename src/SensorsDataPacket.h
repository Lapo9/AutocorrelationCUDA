#ifndef AUTOCORRELATIONCUDA_SENSORSDATAPACKET
#define AUTOCORRELATIONCUDA_SENSORSDATAPACKET

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <cmath>
#include <iostream>

#include "Definitions.h"


namespace AutocorrelationCUDA {

/**
* @brief Packet of data received from the sensors, stored on GPU.
* @details Data is organized following this layout:
* 
*    sens.0  sens.1  sens.2  sens. ...
	+-------+-------+-------+-------
	|  inst |  inst |  inst |  ...
	|  0    |  0    |  0    |
	+-------+-------+-------+-------
	|  inst |  inst |  inst |  ...
	|  1    |  1    |  1    |
	+-------+-------+-------+-------
	|  inst |  inst |  inst |  ...
	|  2    |  2    |  2    |
	+-------+-------+-------+-------
	|  inst |  inst |  inst |  ...
	|  3    |  3    |  3    |
	+-------+-------+-------+-------
	|  ...  |  ...  |  ...  |  ...
	|       |       |       |

**/
class SensorsDataPacket final {
	
	public:

	/**
	* @brief Creates the array containing the data on GPU, and initializes it with 0.
	**/
	__host__ SensorsDataPacket() {
		
		std::cout << "\ninitializing SensorsDataPacket...\n";

		cudaMalloc(&data, SENSORS * INSTANTS_PER_PACKET * sizeof(uint8));
		cudaMemset(data, 0, SENSORS * INSTANTS_PER_PACKET);

		std::cout << "\nSensorsDataPacket done!\n";
	}



	/**
	* @brief Returns the value for the specified sensor and instant.
	* @param sensor The sensor.
	* @param instant The instant.
	* @return The value for the specified sensor and instant.
	* @pre sensor < SENSORS; instant < INSTANTS_PER_PACKET
	**/
	__device__ uint8 get(uint16 sensor, uint16 instant) {
		return data[instant * SENSORS + sensor];
	}



	/**
	* @brief Uploads a new data packet to the GPU.
	* @param newData Reference to the new packet to be uploaded.
	**/
	__host__ void setNewDataPacket(const std::vector<uint8>& newData) {
		cudaMemcpy(data, newData.data(), SENSORS * INSTANTS_PER_PACKET * sizeof(uint8), cudaMemcpyHostToDevice);
	}




	private:

	uint8* data;

};
}


#endif

