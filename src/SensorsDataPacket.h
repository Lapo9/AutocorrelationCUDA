#ifndef AUTOCORRELATIONCUDA_SENSORSDATAPACKET
#define AUTOCORRELATIONCUDA_SENSORSDATAPACKET

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <cmath>


namespace AutocorrelationCUDA {

template <typename Contained>
class SensorsDataPacket final {
	
	public:

	__host__ SensorsDataPacket(unsigned int sensorsExp2, unsigned int instants) {
		this->sensors = std::pow(2, sensorsExp2);
		this->instants = instants;

		cudaMalloc(&arr, sensors * instants);
		cudaMemset(arr, 0, sensors * instants);
		
		Contained tmp[2] = {sensors, instants};
		cudaMemcpy(arr, tmp, 2 * sizeof(Contained), cudaMemcpyHostToDevice);
	}



	__device__ Contained get(std::uint_fast8_t sensor, std::uint_fast8_t instant) {
		return arr[2 + instant * sensorsNum() + sensor];
	}



	__host__ void setNewDataPacket(const std::vector<Contained>& newData) {
		cudaMemcpy(arr+2, newData.data(), sensors * instants, cudaMemcpyHostToDevice);
	}



	__device__ Contained sensorsNum() {
		return arr[0];
	}


	__device__ Contained instantsNum() {
		return arr[1];
	}


	private:

	Contained* arr;

	Contained sensors;
	Contained instants;

};
}


#endif

