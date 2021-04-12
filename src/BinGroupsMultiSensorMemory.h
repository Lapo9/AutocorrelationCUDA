#ifndef AUTOCORRELATIONCUDA_BINGROUPSMULTISENSORMEMORY
#define AUTOCORRELATIONCUDA_BINGROUPSMULTISENSORMEMORY

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>




/*
* 
	+-----------+-----------+-----------+-----------+
	|	group	|	num of	|	num of	|cells per	|
	|	size	|	groups	|	sensors	|	sensor	|
	+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
s.0	|	zero	|	zero	|	zero	|	...		|accumulator|	gr.0	|	gr.0	|	...		|accumulator|	...		|accumulator|	gr.1	|	gr.1	|	...		|accumulator|	...		|	...
-->	|delay gr.0	|delay gr.1	|delay gr.2	|			|	pos gr.0|	pos x	|	pos x+1	|			|	gr.0	|			|	pos gr.1|	pos x	|	pos x+1	|			|	gr.1	|			|
	+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
s.1	|	zero	|	zero	|	zero	|	...		|accumulator|	gr.0	|	gr.0	|	...		|accumulator|	...		|accumulator|	gr.1	|	gr.1	|	...		|accumulator|	...		|	...
-->	|delay gr.0	|delay gr.1	|delay gr.2	|			|	pos gr.0|	pos x	|	pos x+1	|			|	gr.0	|			|	pos gr.1|	pos x	|	pos x+1	|			|	gr.1	|			|
	+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
s.2	|	zero	|	zero	|	zero	|	...		|accumulator|	gr.0	|	gr.0	|	...		|accumulator|	...		|accumulator|	gr.1	|	gr.1	|	...		|accumulator|	...		|	...
-->	|delay gr.0	|delay gr.1	|delay gr.2	|			|	pos gr.0|	pos x	|	pos x+1	|			|	gr.0	|			|	pos gr.1|	pos x	|	pos x+1	|			|	gr.1	|			|
	+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
s.3	|	zero	|	zero	|	zero	|	...		|accumulator|	gr.0	|	gr.0	|	...		|accumulator|	...		|accumulator|	gr.1	|	gr.1	|	...		|accumulator|	...		|	...
-->	|delay gr.0	|delay gr.1	|delay gr.2	|			|	pos gr.0|	pos x	|	pos x+1	|			|	gr.0	|			|	pos gr.1|	pos x	|	pos x+1	|			|	gr.1	|			|
	+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
...	|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...
-->	|			|			|			|			|			|			|			|			|			|			|			|			|			|			|			|			|
	
*/



template <typename Contained, int SizeExp>
class BinGroupsMultiSensorMemory {

	public:

	/**
	 * @brief Returns the i-th value of the binGroup-th group of the sensor-th sensor
	 * @param sensor Sensor to return the specifeid value of
	 * @param binGroup Group of the specified sensor to return the i-th value
	 * @param i i-th value of the specified group of the specified sensor
	 * @return The i-th value of the binGroup-th group of the sensor-th sensor
	 * @pre \p sensor < sensorsNum(), \p binGroup < groupsNum(), \p i < groupSize() - 1
	*/
	__device__ public Contained get(std::uint_fast8_t sensor, std::uint_fast8_t binGroup, std::uint_fast8_t i) {
		std::uint_fast8_t startOfGroup = getAccumulatorPosPos(); //place where there is the position of the accumulator
		int pos : SizeExp {arr[startOfGroup] + 1 + i};
		return arr[startOfGroup + 1 + pos];
	}



	//TODO it is possible to do fewer calculus, indeed all of the functions called redo the call to getAccumulatorPosPos
	__device__ void shift(std::uint_fast8_t sensor, std::uint_fast8_t binGroup) {
		decrementAccumulatorPos();
		if (binGroup < groupsNum() - 1) {
			addToAccumulator(sensor, binGroup+1, getAccumulator(sensor, binGroup)); //after the decrement, currentAccumulator is the place where there is the expelled value
		}
		clearAccumulator(sensor, binGroup);
	}



	__device__ public Contained getAccumulator(std::uint_fast8_t sensor, std::uint_fast8_t binGroup) {
		std::uint_fast8_t startOfGroup = getAccumulatorPosPos(); //place where there is the position of the accumulator
		return arr[startOfGroup+1 + arr[startOfGroup]];
	}



	__device__ public Contained getZeroDelay(std::uint_fast8_t sensor, std::uint_fast8_t binGroup) {
		return arr[4 + sensor * cellsPerSensor() + binGroup];
	}




	private:

	__device__ Contained groupSize() {
		return arr[0];
	}

	__device__ Contained groupsNum() {
		return arr[1];
	}

	__device__ Contained sensorsNum() {
		return arr[2];
	}

	__device__ Contained cellsPerSensor() {
		arr[3]
	}

	__device__ Contained getAccumulatorPosPos(std::uint_fast8_t sensor, std::uint_fast8_t binGroup) {
		//TODO possible proxy, but is it really faster to compare sensor and binGroup to the previous ones (possibly stored as 5th and 6th elements in the array
		return 4 + sensor * cellsPerSensor() + sensorsNum() + binGroup * groupSize();
	}

	__device__ void decrementAccumulatorPos(std::uint_fast8_t sensor, std::uint_fast8_t binGroup) {
		std::uint_fast8_t startOfGroup = getAccumulatorPosPos(); //place where there is the position of the accumulator
		int newPos : SizeExp {arr[startOfGroup] - 1};
		arr[startOfGroup] = newPos;
	}



	__device__ public void addToAccumulator(std::uint_fast8_t sensor, std::uint_fast8_t binGroup, Contained add) {
		std::uint_fast8_t startOfGroup = getAccumulatorPosPos(); //place where there is the position of the accumulator
		arr[startOfGroup + 1 + arr[startOfGroup]] += add;
	}



	__device__ public void clearAccumulator(std::uint_fast8_t sensor, std::uint_fast8_t binGroup) {
		std::uint_fast8_t startOfGroup = getAccumulatorPosPos(); //place where there is the position of the accumulator
		arr[startOfGroup + 1 + arr[startOfGroup]] = 0;
	}




	Contained* arr;
	

};


#endif
