#ifndef AUTOCORRELATIONCUDA_BINGROUPSMULTISENSORMEMORY
#define AUTOCORRELATIONCUDA_BINGROUPSMULTISENSORMEMORY

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <cmath>




/*
 
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


namespace AutocorrelationCUDA {

template <typename Contained, int SizeExp2>
class BinGroupsMultiSensorMemory final {

	public:

	__host__ BinGroupsMultiSensorMemory(Contained sensors, Contained groups) {
		this->groupSizev = std::pow(2, SizeExp2)-1;
		this->sensors = sensors;
		this->groups = groups;

		Contained cellsPerSensor = groups + groups * (groupSizev + 2);
		std::size_t totalCells = 4 + cellsPerSensor * sensors;

		Contained tmp[4] = {groupSizev, groups, sensors, cellsPerSensor};

		cudaMalloc(&arr, totalCells * sizeof(Contained));
		cudaMemset(arr, 0, totalCells * sizeof(Contained));
		cudaMemcpy(arr, tmp, 4 * sizeof(Contained), cudaMemcpyHostToDevice);
	}




	__host__ ResultArray<Contained> generateResultArray() {
		return ResultArray<Contained>(sensors, groups * groupSizev);
	}



	/**
	 * @brief Returns the i-th value of the binGroup-th group of the sensor-th sensor
	 * @param sensor Sensor to return the specifeid value of
	 * @param binGroup Group of the specified sensor to return the i-th value
	 * @param i i-th value of the specified group of the specified sensor
	 * @return The i-th value of the binGroup-th group of the sensor-th sensor
	 * @pre \p sensor < sensorsNum(), \p binGroup < groupsNum(), \p i < groupSize() - 1
	*/
	__device__ Contained get(std::uint_fast16_t sensor, std::uint_fast16_t binGroup, std::uint_fast16_t i) {
		std::uint_fast16_t startOfGroup = getAccumulatorPosPos(sensor, binGroup); //place where there is the position of the accumulator
		SizeExpModuleMathUnsignedInt pos;
		pos.bitfield = arr[startOfGroup] + 1 + i;
		return arr[startOfGroup + 1 + pos.bitfield];
	}



	//TODO it is possible to do fewer calculus, indeed all of the functions called redo the call to getAccumulatorPosPos
	__device__ void shift(std::uint_fast16_t sensor, std::uint_fast16_t binGroup) {
		decrementAccumulatorPos(sensor, binGroup);

		//if there is next bin group
		if (binGroup < groupsNum() - 1) {
			addToAccumulator(sensor, binGroup+1, getAccumulator(sensor, binGroup)); //after the decrement, currentAccumulator is the place where there is the expelled value
			addToZeroDelay(sensor, binGroup+1, getZeroDelay(sensor, binGroup)); //add current zero delay to the next one
		}

		//clear current
		clearAccumulator(sensor, binGroup);
		clearZeroDelay(sensor, binGroup);
	}



	__device__ Contained getZeroDelay(std::uint_fast16_t sensor, std::uint_fast16_t binGroup) {
		return arr[4 + sensor * cellsPerSensor() + binGroup];
	}



	__device__ void insertNew(std::uint_fast16_t sensor, Contained datum) {
		addToAccumulator(sensor, 0, datum);
		addToZeroDelay(sensor, 0, datum);
	}



	__device__ Contained groupsNum() {
		return arr[1];
	}

	__device__ Contained sensorsNum() {
		return arr[2];
	}



	private:

	__device__ void decrementAccumulatorPos(std::uint_fast16_t sensor, std::uint_fast16_t binGroup) {
		std::uint_fast16_t startOfGroup = getAccumulatorPosPos(sensor, binGroup); //place where there is the position of the accumulator
		SizeExpModuleMathUnsignedInt newPos;
		newPos.bitfield = arr[startOfGroup] - 1; //does it really work?
		arr[startOfGroup] = newPos.bitfield;
	}



	__device__ Contained getAccumulator(std::uint_fast16_t sensor, std::uint_fast16_t binGroup) {
		std::uint_fast16_t startOfGroup = getAccumulatorPosPos(sensor, binGroup); //place where there is the position of the accumulator
		return arr[startOfGroup + 1 + arr[startOfGroup]];
	}



	__device__ void addToAccumulator(std::uint_fast16_t sensor, std::uint_fast16_t binGroup, Contained add) {
		std::uint_fast16_t startOfGroup = getAccumulatorPosPos(sensor, binGroup); //place where there is the position of the accumulator
		arr[startOfGroup + 1 + arr[startOfGroup]] += add;
	}



	__device__ void clearAccumulator(std::uint_fast16_t sensor, std::uint_fast16_t binGroup) {
		std::uint_fast16_t startOfGroup = getAccumulatorPosPos(sensor, binGroup); //place where there is the position of the accumulator
		arr[startOfGroup + 1 + arr[startOfGroup]] = 0;
	}



	__device__ void addToZeroDelay(std::uint_fast16_t sensor, std::uint_fast16_t binGroup, Contained add) {
		arr[4 + sensor * groupSize() + binGroup] += add;
	}



	__device__ void clearZeroDelay(std::uint_fast16_t sensor, std::uint_fast16_t binGroup) {
		arr[4 + sensor * groupSize() + binGroup] = 0;
	}



	__device__ Contained groupSize() {
		return arr[0];
	}

	__device__ Contained cellsPerSensor() {
		return arr[3];
	}

	__device__ Contained getAccumulatorPosPos(std::uint_fast16_t sensor, std::uint_fast16_t binGroup) {
		//TODO possible proxy, but is it really faster to compare sensor and binGroup to the previous ones (possibly stored as 5th and 6th elements in the array
		return 4 + sensor * cellsPerSensor() + sensorsNum() + binGroup * groupSize();
	}



	Contained* arr;

	Contained sensors;
	Contained groups;
	Contained groupSizev;

	struct SizeExpModuleMathUnsignedInt {
		unsigned int bitfield : SizeExp2;
	};

};

}

#endif
