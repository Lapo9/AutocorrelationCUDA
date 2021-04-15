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

		cudaMalloc(&data, sensors * instants * sizeof(Contained));
		cudaMemset(data, 0, sensors * instants);
		
		std::uint_fast32_t tmp[2] = {sensors, instants};
		cudaMalloc(&info, 2 * sizeof(std::uint_fast32_t));
		cudaMemcpy(info, tmp, 2 * sizeof(std::uint_fast32_t), cudaMemcpyHostToDevice);
	}



	__device__ Contained get(std::uint_fast32_t sensor, std::uint_fast32_t instant) {
		return data[instant * sensorsNum() + sensor];
	}



	__host__ void setNewDataPacket(const std::vector<Contained>& newData) {
		cudaMemcpy(data, newData.data(), sensors * instants * sizeof(Contained), cudaMemcpyHostToDevice);
	}



	__device__ std::uint_fast32_t sensorsNum() {
		return info[0];
	}


	__device__ std::uint_fast32_t instantsNum() {
		return info[1];
	}


	private:

	Contained* data;
	std::uint_fast32_t* info;

	std::uint_fast32_t sensors;
	std::uint_fast32_t instants;


};
}


#endif

