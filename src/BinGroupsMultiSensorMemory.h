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
			+-----------+-----------+-----------+-----------+


			+-----------+-----------+-----------+-------
	sens.0	|	zero	|	zero	|	zero	|	...	
	-->		|delay gr.0	|delay gr.1	|delay gr.2	|		
			+-----------+-----------+-----------+-------
	sens.1	|	zero	|	zero	|	zero	|	...	
	-->		|delay gr.0	|delay gr.1	|delay gr.2	|		
			+-----------+-----------+-----------+-------
	sens.2	|	zero	|	zero	|	zero	|	...	
	-->		|delay gr.0	|delay gr.1	|delay gr.2	|		
			+-----------+-----------+-----------+-------
	sens.3	|	zero	|	zero	|	zero	|	...	
	-->		|delay gr.0	|delay gr.1	|delay gr.2	|		
			+-----------+-----------+-----------+-------
	...		|	...		|	...		|	...		|	...	
	-->		|			|			|			|		



			+-----------+-----------+-----------+-------
	sens.0	|accumulator|accumulator|accumulator|	...
	-->		|	gr.0	|	gr.1	|	gr.2	|
			+-----------+-----------+-----------+-------
	sens.1	|accumulator|accumulator|accumulator|	...
	-->		|	gr.0	|	gr.1	|	gr.2	|
			+-----------+-----------+-----------+-------
	sens.2	|accumulator|accumulator|accumulator|	...
	-->		|	gr.0	|	gr.1	|	gr.2	|
			+-----------+-----------+-----------+-------
	sens.3	|accumulator|accumulator|accumulator|	...
	-->		|	gr.0	|	gr.1	|	gr.2	|
			+-----------+-----------+-----------+-------
	...		|	...		|	...		|	...		|	...
	-->		|			|			|			|



			+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
	sens.0	|	gr.0	|	gr.0	|	...		|accumulator|	...		|accumulator|	gr.1	|	gr.1	|	...		|accumulator|	...		|	...
	-->		|	pos x	|	pos x+1	|			|	gr.0	|			|	pos gr.1|	pos x	|	pos x+1	|			|	gr.1	|			|
			+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
	sens.1	|	gr.0	|	gr.0	|	...		|accumulator|	...		|accumulator|	gr.1	|	gr.1	|	...		|accumulator|	...		|	...
	-->		|	pos x	|	pos x+1	|			|	gr.0	|			|	pos gr.1|	pos x	|	pos x+1	|			|	gr.1	|			|
			+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
	sens.2	|	gr.0	|	gr.0	|	...		|accumulator|	...		|accumulator|	gr.1	|	gr.1	|	...		|accumulator|	...		|	...
	-->		|	pos x	|	pos x+1	|			|	gr.0	|			|	pos gr.1|	pos x	|	pos x+1	|			|	gr.1	|			|
			+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
	sens.3	|	gr.0	|	gr.0	|	...		|accumulator|	...		|accumulator|	gr.1	|	gr.1	|	...		|accumulator|	...		|	...
	-->		|	pos x	|	pos x+1	|			|	gr.0	|			|	pos gr.1|	pos x	|	pos x+1	|			|	gr.1	|			|
			+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------
	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...		|	...
	-->		|			|			|			|			|			|			|			|			|			|			|			|
		
*/


namespace AutocorrelationCUDA {

template <typename Contained, int SizeExp2>
class BinGroupsMultiSensorMemory final {

	public:

	__host__ BinGroupsMultiSensorMemory(Contained sensors, Contained groups) {
		this->groupSizev = std::pow(2, SizeExp2)-1;
		this->sensors = sensors;
		this->groups = groups;

		std::uint_fast32_t cellsPerSensor = groups * (groupSizev + 1); //TODO

		std::uint_fast32_t tmp[4] = {groupSizev, groups, sensors, cellsPerSensor};

		//create matrix for data on GPU and fill it with 0
		cudaMalloc(&data, cellsPerSensor * sensors * sizeof(Contained));
		cudaMemset(data, 0, cellsPerSensor * sensors * sizeof(Contained));

		//create matrix for zero delay data on GPU and fill it with 0
		cudaMalloc(&zeroDelays, groups * sensors * sizeof(Contained));
		cudaMemset(zeroDelays, 0, groups * sensors * sizeof(Contained));

		//create matrix for accumulator positions for each group on GPU and fill it with 0
		cudaMalloc(&accumulatorsPos, groups * sensors * sizeof(Contained));
		cudaMemset(accumulatorsPos, 0, groups * sensors * sizeof(Contained));

		//create array for info on GPU and fill it
		cudaMalloc(&info, 4 * sizeof(std::uint_fast32_t));
		cudaMemcpy(info, tmp, 4 * sizeof(std::uint_fast32_t), cudaMemcpyHostToDevice);
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
	__device__ Contained get(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast32_t i) {
		std::uint_fast8_t accumulatorRelativePos = getAccumulatorRelativePos(sensor, binGroup);
		std::uint_fast32_t startOfGroup = getStartOfGroup(sensor, binGroup);
		SizeExpModuleMathUnsignedInt pos;
		pos.bitfield = accumulatorRelativePos + 1 + i;
		return data[startOfGroup + pos.bitfield];
	}



	//TODO it is possible to do fewer calculus, indeed all of the functions called redo the call to getAccumulatorPosPos
	__device__ void shift(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
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



	__device__ Contained getZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		return zeroDelays[sensor * groupsNum() + binGroup];
	}



	__device__ void insertNew(std::uint_fast32_t sensor, Contained datum) {
		addToAccumulator(sensor, 0, datum);
		addToZeroDelay(sensor, 0, datum);
	}



	__device__ std::uint_fast32_t groupsNum() {
		return info[1];
	}

	__device__ std::uint_fast32_t sensorsNum() {
		return info[2];
	}



	private:

	__device__ void decrementAccumulatorPos(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		std::uint_fast32_t accumulatorPos = getAccumulatorRelativePos(sensor, binGroup);
		SizeExpModuleMathUnsignedInt newPos;
		newPos.bitfield = accumulatorsPos - 1; //does it really work?
		accumulatorsPos[accumulatorPos] = newPos.bitfield;
	}



	__device__ Contained getAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		return data[getStartOfGroup(sensor, binGroup) + getAccumulatorRelativePos(sensor, binGroup)];
	}



	__device__ void addToAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained add) {
		data[getStartOfGroup(sensor, binGroup) + getAccumulatorRelativePos(sensor, binGroup)] += add;
	}



	__device__ void clearAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		data[getStartOfGroup(sensor, binGroup) + getAccumulatorRelativePos(sensor, binGroup)] = 0;
	}



	__device__ void addToZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained add) {
		zeroDelays[sensor * groupsNum() + binGroup] += add;
	}



	__device__ void clearZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		zeroDelays[sensor * groupsNum() + binGroup] = 0;
	}



	__device__ std::uint_fast32_t groupSize() {
		return info[0];
	}

	__device__ std::uint_fast32_t cellsPerSensor() {
		return info[3];
	}

	__device__ std::uint_fast32_t getAccumulatorRelativePos(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		return accumulatorsPos[sensor * groupsNum() + binGroup];
	}

	__device__ std::uint_fast32_t getStartOfGroup(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		retun sensor * cellsPerSensor() + binGroup * groupSize();
	}



	Contained* data;
	Contained* zeroDelays;
	std::uint8_t* accumulatorsPos;
	std::uint_fast32_t* info;

	std::uint_fast32_t sensors;
	std::uint_fast32_t groups;
	std::uint_fast32_t groupSizev;

	struct SizeExpModuleMathUnsignedInt {
		unsigned int bitfield : SizeExp2;
	};

};

}

#endif
