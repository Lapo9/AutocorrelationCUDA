#ifndef AUTOCORRELATIONCUDA_RESULTARRAY
#define AUTOCORRELATIONCUDA_RESULTARRAY


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>




/*

			+-------+-------+
			|	max	|sens-	|
			|	lag	|	ors	|
			+-------+-------+

			+-------+-------+-------+-------+-------+----------
sensor 0	|	lag	|	lag	|	lag	|	lag	|	lag	|	lag	
	-->		|	1	|	2	|	3	|	4	|	5	|	...	
			+-------+-------+-------+-------+-------+----------
sensor 1	|	lag	|	lag	|	lag	|	lag	|	lag	|	lag
	-->		|	1	|	2	|	3	|	4	|	5	|	...
			+-------+-------+-------+-------+-------+----------
sensor 2	|	lag	|	lag	|	lag	|	lag	|	lag	|	lag
	-->		|	1	|	2	|	3	|	4	|	5	|	...
			+-------+-------+-------+-------+-------+----------
sensor 3	|	lag	|	lag	|	lag	|	lag	|	lag	|	lag
	-->		|	1	|	2	|	3	|	4	|	5	|	...
			+-------+-------+-------+-------+-------+----------
	...		|	...	|	...	|	...	|	...	|	...	|	...	
	-->		|		|		|		|		|		|		

*/


namespace AutocorrelationCUDA {

template <typename Contained>
class ResultArray final {
	
	public:

	__host__ ResultArray(std::uint_fast32_t sensors, std::uint_fast32_t maxLag) {
		this->sensorsv = sensors;
		this->maxLagv = maxLag;

		 cudaMalloc(&data, sensors * maxLag * sizeof(Contained));
		 cudaMemset(data, 0, (1 + sensors * maxLag) * sizeof(Contained));

		 cudaMalloc(&info, 2 * sizeof(std::uint_fast32_t));
		 std::uint_fast32_t tmp[2] = {maxLag, sensors};
		 cudaMemcpy(info, tmp, 2 * sizeof(std::uint_fast32_t), cudaMemcpyHostToDevice);
	}


	__device__ void addTo(std::uint_fast32_t sensor, std::uint_fast32_t lag, Contained datum) {
		data[sensor * maxLag() + lag] += datum;
	}


	__host__ Contained get(std::uint_fast32_t sensor, std::uint_fast32_t lag) {
		return toVector()[sensor * maxLagv + lag];
	}


	__host__ std::vector<Contained> toVector() {		
		std::vector<Contained> result(maxLagv * sensorsv);
		cudaMemcpy(result.data(), data, maxLagv * sensorsv * sizeof(Contained), cudaMemcpyDeviceToHost);
		return result;
	}


	__host__ std::uint_fast32_t getMaxLagv() {
		return maxLagv;
	}


	__host__ std::uint_fast32_t getSensors() {
		return sensorsv;
	}



	private:

	__device__ std::uint_fast32_t maxLag() {
		return info[0];
	}


	__device__ std::uint_fast32_t sensors() {
		return info[1];
	}



	Contained* data;
	std::uint_fast32_t* info;

	std::uint_fast32_t sensorsv;
	std::uint_fast32_t maxLagv;

};
}


#endif
