#ifndef AUTOCORRWLATIONCUDA_BINSTRUCTURE
#define AUTOCORRWLATIONCUDA_BINSTRUCTURE

#include "BinGroup.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace AutocorrelationCUDA{

template <typename Type, int SizeExp>
class BinStructure {

	public:

	__host__ BinStructure(int groups) {
		
	}
	

		
	//shift and add to nex accumulator
	__device__ void shift(int bin) {
		Type tmp = arr[bin].shift();
		arr[bin].addToAccumulator(tmp);
	}


	__device__ void addFirst(Type datum) {
		arr[0].addToAccumulator(datum);
	}


	__device__ void addToZeroDelay(int bin, Type datum) {
		arr[bin].addToZeroDelay(datum);
	}


	__device__ Type getZeroDelay(int bin) {
		return arr[bin].getZeroDelay();
	}


	__device__ void clearZeroDelay(int bin) {
		arr[bin].clearZeroDelay();
	}


	private:

	BinGroup<Type, SizeExp>* arr;

};
}


#endif


