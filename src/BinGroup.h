#ifndef AUTOCORRELATIONCUDA_BINGROUP
#define AUTOCORRELATIONCUDA_BINGROUP

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace AutocorrelationCUDA{

template <typename Type, int SizeExp>
class BinGroup final {

	public:

	__host__ BinGroup() {
		cudaMalloc(&arr, pow(2, SizeExp) * sizeof(Type));
	}



	__device__ void addToAccumulator(Type datum) {
		arr[accumulatorIndex] += datum;
	}


	__device__ Type shift() {
		accumulatorIndex--;
	}



	__device__ void addToZeroDelay(Type datum) {
		zeroDelay += datum;
	}


	__device__ Type getZeroDelay() {
		return zeroDelay;
	}


	__device__ void clearZeroDelay() {
		zeroDelay = 0;
	}


	__device__ Type operator[](int pos) {
		int i : SizeExp {accumulatorIndex+1};
		return arr[i];
	}



	private:

	Type* arr;
	int accumulatorIndex : SizeExp {0};
	Type zeroDelay = 0;
};
}

#endif

