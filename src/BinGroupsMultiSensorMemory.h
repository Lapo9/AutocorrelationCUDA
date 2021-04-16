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



	next array structure eliminates banks conflict in CUDA shared memory
	k = #banks / #banksForGroup -1
		#banksForGroup = (groupSize * datumBytesSize) / #bytesInBank
		#banks = 32; #bytesInBank = 8; datumBytesSize = sizeof(Contained) [= 1]
	==> threadsPerBlock = k * groupSize

																									
																									
	 __	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------++
	|	|	s.0 g.0	|	s.0 g.0	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		||
	|	|	pos x	|	pos x+1	|			|	s.0 g.0	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			||
	|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------++
	|	|	s.0 g.1	|	s.0 g.1	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		||
	|	|	pos x	|	pos x+1	|			|	s.0 g.1	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			||
	|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------++
blk	|	|s.0 g.2	|	s.0 g.2	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		||
0	|	|	pos x	|	pos x+1	|			|	s.0 g.2	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			||
	|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------++
	|	|	...		|	...		|	...		|	...		|	...		||	...		|	...		|	...		|	...		|	...		||	...		||	...		|	...		|	...		|	...		|	...		||
	|	|			|			|			|			|			||			|			|			|			|			||			||			|			|			|			|			||
	|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------++
	|	|s.k+1 g.0	|	s.0 g.0	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		||
	|__	|	pos x	|	pos x+1	|			|	s.0 g.0	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			||
	 __	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------++
	|	|	s.0 g.1	|	s.0 g.1	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		||
	|	|	pos x	|	pos x+1	|			|	s.0 g.1	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			||
	|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------++
blk	|	|s.0 g.2	|	s.0 g.2	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		||
1	|	|	pos x	|	pos x+1	|			|	s.0 g.2	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			||
	|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------++
	|	|	...		|	...		|	...		|	...		|	...		||	...		|	...		|	...		|	...		|	...		||	...		||	...		|	...		|	...		|	...		|	...		||
	|	|			|			|			|			|			||			|			|			|			|			||			||			|			|			|			|			||	
	|	
*/


namespace AutocorrelationCUDA {

//TODO maybe pack pointers arguments in methods signatures into struct? Any downside in performance?
//FIXME change access methods based on new memory layout
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
	__device__ Contained get(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast32_t i, Contained* accsPos, Contained* data, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		std::uint_fast8_t accumulatorRelativePos = getAccumulatorRelativePos(sensor, binGroup, accsPos, accumulatorsPos, info);
		std::uint_fast32_t startOfGroup = getStartOfGroup(sensor, binGroup, info);
		SizeExpModuleMathUnsignedInt pos;
		pos.bitfield = accumulatorRelativePos + 1 + i;
		return data[startOfGroup + pos.bitfield];
	}



	//TODO it is possible to do fewer calculus, indeed all of the functions called redo the call to getAccumulatorPosPos
	__device__ void shift(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* data, Contained* zeroDelays, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		decrementAccumulatorPos(sensor, binGroup, accumulatorsPos, info);

		//if there is next bin group
		if (binGroup < groupsNum(info) - 1) {
			addToAccumulator(sensor, binGroup+1, getAccumulator(sensor, binGroup, data, accumulatorsPos, info), data, accumulatorsPos, info); //after the decrement, currentAccumulator is the place where there is the expelled value
			addToZeroDelay(sensor, binGroup+1, getZeroDelay(sensor, binGroup, zeroDelays, info), zeroDelays, info); //add current zero delay to the next one
		}

		//clear current
		clearAccumulator(sensor, binGroup, data, accumulatorsPos, info);
		clearZeroDelay(sensor, binGroup, zeroDelays, info);
	}



	__device__ Contained getZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* zeroDelays, std::uint_fast32_t* info) {
		return zeroDelays[sensor * groupsNum(info) + binGroup];
	}



	__device__ void insertNew(std::uint_fast32_t sensor, Contained datum, Contained* data, Contained* zeroDelays, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		addToAccumulator(sensor, 0, datum, data, accumulatorsPos, info);
		addToZeroDelay(sensor, 0, datum, zeroDelays, info);
	}



	__device__ std::uint_fast32_t groupsNum(std::uint_fast32_t* info) {
		return info[1];
	}

	__device__ std::uint_fast32_t sensorsNum(std::uint_fast32_t* info) {
		return info[2];
	}



	private:

	__device__ void decrementAccumulatorPos(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		std::uint_fast32_t accumulatorPos = getAccumulatorRelativePos(sensor, binGroup, accumulatorsPos, info);
		SizeExpModuleMathUnsignedInt newPos;
		newPos.bitfield = accumulatorsPos - 1; //does it really work?
		accumulatorsPos[accumulatorPos] = newPos.bitfield;
	}



	__device__ Contained getAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* data, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		return data[getStartOfGroup(sensor, binGroup, info) + getAccumulatorRelativePos(sensor, binGroup, accumulatorsPos, info)];
	}



	__device__ void addToAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained add, Contained* data, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		data[getStartOfGroup(sensor, binGroup, info) + getAccumulatorRelativePos(sensor, binGroup, accumulatorsPos, info)] += add;
	}



	__device__ void clearAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* data, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		data[getStartOfGroup(sensor, binGroup, info) + getAccumulatorRelativePos(sensor, binGroup, accumulatorsPos, info)] = 0;
	}



	__device__ void addToZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained add, Contained* zeroDelays, std::uint_fast32_t* info) {
		zeroDelays[sensor * groupsNum(info) + binGroup] += add;
	}



	__device__ void clearZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* zeroDelays, std::uint_fast32_t* info) {
		zeroDelays[sensor * groupsNum(info) + binGroup] = 0;
	}



	//TODO new layout
	__device__ std::uint_fast32_t getStartOfGroup(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast32_t* info) {
		return binGroup * groupSize(info) + sensor * cellsPerSensor(info);
	}




	__device__ std::uint_fast32_t groupSize(std::uint_fast32_t* info) {
		return info[0];
	}

	__device__ std::uint_fast32_t cellsPerSensor(std::uint_fast32_t* info) {
		return info[3];
	}

	__device__ std::uint_fast32_t getAccumulatorRelativePos(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		return accumulatorsPos[sensor * groupsNum(info) + binGroup];
	}


	//arrays in GPU global memory
	Contained* data;
	Contained* zeroDelays;
	std::uint8_t* accumulatorsPos;
	std::uint_fast32_t* info;

	//values fo host
	std::uint_fast32_t sensors;
	std::uint_fast32_t groups;
	std::uint_fast32_t groupSizev;



	struct SizeExpModuleMathUnsignedInt {
		unsigned int bitfield : SizeExp2;
	};

};

}

#endif
