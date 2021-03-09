#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Feeder.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;


int main() {

	AutocorrelationCUDA::Feeder f1{2s, [] {std::cout << "Ciao!\n" << std::endl; }};
	f1.start();

	std::this_thread::sleep_for(4s);
	f1.pause();
	std::this_thread::sleep_for(6s);
	f1.resume();

	std::this_thread::sleep_for(10s);
	//receive data
	//send data to GPU
	//launch kernel
	//loop

	//collect results

}


/*__global__ void autocorrelate(int maxLag, float* lagStart, float* start, int length) {

	int pivotStart = 0;
	for (int i = 0; i < start - lagStart + length; ++i) {

		pivotStart = (lagStart+i <= start ? 0 : lagStart+i-start);
		if(start+threadIdx.x-(lagStart+i) <= maxLag) {
			(*(lagStart+i)) * (*(start+pivotStart+threadIdx.x));
		}
	}

}*/