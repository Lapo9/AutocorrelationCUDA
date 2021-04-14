#ifndef AUTOCORRELATIONCUDA_RESULTARRAY
#define AUTOCORRELATIONCUDA_RESULTARRAY


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>




/*

			+-------+
			|	max	|
			|	lag	|
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

	__host__ ResultArray(std::uint_fast16_t sensors, Contained maxLag) {
		this->sensors = sensors;
		this->maxLagv = maxLag;

		 cudaMalloc(&arr, (1 + sensors * maxLag) * sizeof(Contained));
		 cudaMemset(arr, 0, (1 + sensors * maxLag) * sizeof(Contained));
		 cudaMemcpy(arr, &maxLag, sizeof(Contained), cudaMemcpyHostToDevice);
	}


	__device__ void addTo(std::uint_fast16_t sensor, std::uint_fast16_t lag, Contained datum) {
		arr[1 + sensor * maxLag() + lag] += datum;
	}


	__host__ Contained get(std::uint_fast16_t sensor, std::uint_fast16_t lag) {
		return toVector()[1 + sensor * maxLagv + lag];
	}


	__host__ std::vector<Contained> toVector() {		
		std::vector<Contained> result(maxLagv * sensors);
		cudaMemcpy(result.data(), arr+2, maxLagv * sensors * sizeof(Contained), cudaMemcpyDeviceToHost);
		return result;
	}


	__host__ std::size_t getMaxLagv() {
		return maxLagv;
	}


	__host__ std::size_t getSensors() {
		return sensors;
	}

	private:

	__device__ Contained maxLag() {
		return arr[0];
	}

	Contained* arr;

	std::uint_fast16_t sensors;
	Contained maxLagv;

};
}


#endif
