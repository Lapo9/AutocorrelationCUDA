#ifndef AUTOCORRELATIONCUDA_SENSORSDATAPACKET
#define AUTOCORRELATIONCUDA_SENSORSDATAPACKET

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <cmath>
#include <iostream>

#include "Definitions.h"


namespace AutocorrelationCUDA {


class SensorsDataPacket final {
	
	public:

	__host__ SensorsDataPacket() {
		
		std::cout << "\ninitializing SensorsDataPacket...\n";

		cudaMalloc(&data, SENSORS * INSTANTS_PER_PACKET * sizeof(uint8));
		cudaMemset(data, 0, SENSORS * INSTANTS_PER_PACKET);

		std::cout << "\nSensorsDataPacket done!\n";
	}



	__device__ uint8 get(uint16 sensor, uint16 instant) {
		return data[instant * SENSORS + sensor];
	}



	__host__ void setNewDataPacket(const std::vector<uint8>& newData) {
		cudaMemcpy(data, newData.data(), SENSORS * INSTANTS_PER_PACKET * sizeof(uint8), cudaMemcpyHostToDevice);
	}




	private:

	uint8* data;

};
}


#endif

