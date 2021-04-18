#ifndef AUTOCORRELATIONCUDA_BINGROUPSMULTISENSORMEMORY
#define AUTOCORRELATIONCUDA_BINGROUPSMULTISENSORMEMORY

#define NUMBER_OF_BANKS 32
#define BYTES_PER_BANK 8
#define INFO_AMOUNT 7

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <cmath>
#include <iostream>




/*
 
			+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
			|	group	|	num of	|	num of	|cells per	|cells per	|groups per	|sensors per|
			|	size	|	groups	|	sensors	|	bank	|	block	|	block	|	block	|
			+-----------+-----------+-----------+-----------+-----------+-----------+-----------+



		 __	+-----------+-----------+-----------+-----------+
		|	|zero delay	|zero delay	|	...		|zero delay	|
		|	|s.0 g.0	|s.1 g.0	|			|s.k g.0	|
		|	+-----------+-----------+-----------+-----------+
		|	|zero delay	|zero delay	|	...		|zero delay	|
block	|	|s.0 g.1	|s.1 g.1	|			|s.k g.1	|
0	 <--|	+-----------+-----------+-----------+-----------+
		|	|	...		|	...		|	...		|	...		|
		|	|			|			|			|			|
		|	+-----------+-----------+-----------+-----------+
		|	|zero delay	|zero delay	|	...		|zero delay	|
		|__	|s.0 g.h	|s.1 g.h	|			|s.k g.h	|
		 __	+===========+===========+===========+===========+
		|	|zero delay	|zero delay	|	...		|zero delay	|
		|	|s.k+1 g.0	|s.k+2 g.0	|			|s.k+k g.0	|
block	|	+-----------+-----------+-----------+-----------+
1	 <--|	|zero delay	|zero delay	|	...		|zero delay	|
		|	|s.k+1 g.1	|s.k+2 g.1	|			|s.k+k g.1	|
		|	+-----------+-----------+-----------+-----------+
		|	|	...		|	...		|	...		|	...		|
		|	|			|			|			|			|
		|



		 __	+-----------+-----------+-----------+-----------+
		|	|accumulator|accumulator|	...		|accumulator|
		|	|s.0 g.0	|s.1 g.0	|			|s.k g.0	|
		|	+-----------+-----------+-----------+-----------+
		|	|accumulator|accumulator|	...		|accumulator|
block	|	|s.0 g.1	|s.1 g.1	|			|s.k g.1	|
0	 <--|	+-----------+-----------+-----------+-----------+
		|	|	...		|	...		|	...		|	...		|
		|	|			|			|			|			|
		|	+-----------+-----------+-----------+-----------+
		|	|accumulator|accumulator|	...		|accumulator|
		|__	|s.0 g.h	|s.1 g.h	|			|s.k g.h	|
		 __	+===========+===========+===========+===========+
		|	|accumulator|accumulator|	...		|accumulator|
		|	|s.k+1 g.0	|s.k+2 g.0	|			|s.k+k g.0	|
block	|	+-----------+-----------+-----------+-----------+
1	 <--|	|accumulator|accumulator|	...		|accumulator|
		|	|s.k+1 g.1	|s.k+2 g.1	|			|s.k+k g.1	|
		|	+-----------+-----------+-----------+-----------+
		|	|	...		|	...		|	...		|	...		|
		|	|			|			|			|			|
		|




	next array structure eliminates banks conflict in CUDA shared memory
	k = #banks / #banksForGroup -1
		#banksForGroup = (groupSize * datumBytesSize) / #bytesInBank
		#banks = 32; #bytesInBank = 8; datumBytesSize = sizeof(Contained) [= 1]
	==> threadsPerBlock = k * groupSize

																																											
		 __	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	|	s.0 g.0	|	s.0 g.0	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	|	pos x	|	pos x+1	|			|	s.0 g.0	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	|	s.0 g.1	|	s.0 g.1	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	|	pos x	|	pos x+1	|			|	s.0 g.1	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
block	|	|	s.0 g.2	|	s.0 g.2	|	...		|	accum.	|	...		||	s.1 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||	s.k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
0	 <--|	|	pos x	|	pos x+1	|			|	s.0 g.2	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	|	...		|	...		|	...		|	...		|	...		||	...		|	...		|	...		|	...		|	...		||	...		||	...		|	...		|	...		|	...		|	...		|
		|	|			|			|			|			|			||			|			|			|			|			||			||			|			|			|			|			|
		|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	|	s.0 g.h	|	s.0 g.h	|	...		|	accum.	|	...		||	s.1 g.h	|	s.1 g.h	|	...		|	accum.	|	...		||	...		||	s.k g.h	|	s.k g.h	|	...		|	accum.	|	...		|
		|__	|	pos x	|	pos x+1	|			|	s.0 g.h	|			||	pos x	|	pos x+1	|			|	s.1 g.h	|			||			||	pos x	|	pos x+1	|			|	s.k g.h	|			|
		 __	+===========+===========+===========+===========+===========++==========+===========+===========+===========+===========++==========++==========+===========+===========+===========+===========+
		|	|s.k+1 g.0	|s.k+1 g.0	|	...		|	accum.	|	...		||s.k+2 g.0	|	s.1 g.0	|	...		|	accum.	|	...		||	...		||s.k+k g.0	|	s.k g.0	|	...		|	accum.	|	...		|
		|	|	pos x	|	pos x+1	|			|s.k+1 g.0	|			||	pos x	|	pos x+1	|			|	s.1 g.0	|			||			||	pos x	|	pos x+1	|			|	s.k g.0	|			|
		|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
block	|	|s.k+1 g.1	|s.k+1 g.1	|	...		|	accum.	|	...		||s.k+2 g.1	|s.k+2 g.1	|	...		|	accum.	|	...		||	...		||s.k+k g.1	|s.k+k g.1	|	...		|	accum.	|	...		|
1	 <--|	|	pos x	|	pos x+1	|			|s.k+1 g.1	|			||	pos x	|	pos x+1	|			|s.k+2 g.1	|			||			||	pos x	|	pos x+1	|			|s.k+k g.1	|			|
		|	+-----------+-----------+-----------+-----------+-----------++----------+-----------+-----------+-----------+-----------++----------++----------+-----------+-----------+-----------+-----------+
		|	|	...		|	...		|	...		|	...		|	...		||	...		|	...		|	...		|	...		|	...		||	...		||	...		|	...		|	...		|	...		|	...		|
		|	|			|			|			|			|			||			|			|			|			|			||			||			|			|			|			|			|
		|	

*/


namespace AutocorrelationCUDA {

//TODO maybe pack pointers arguments in methods signatures into struct? Any downside in performance?
template <typename Contained, int SizeExp2>
class BinGroupsMultiSensorMemory final {

	public:

	__host__ BinGroupsMultiSensorMemory(Contained sensors, Contained groups) : groupSizev{(std::uint_fast32_t)std::pow(2, SizeExp2)}, sensors{sensors}, groups{groups} {
		
		std::cout << "\ninitializing BinGroupMultiSensor...\n";

		sensorsPerBlockv = NUMBER_OF_BANKS / ((groupSizev * sizeof(Contained)) / BYTES_PER_BANK); 
		std::uint_fast32_t cellsPerBank = sensorsPerBlockv * groupSizev;
		std::uint_fast32_t groupsPerBlock = groups * sensorsPerBlockv;
		std::uint_fast32_t cellsPerBlock = cellsPerBank * groups;

		std::uint_fast32_t tmp[INFO_AMOUNT] = {groupSizev, groups, sensors, cellsPerBank, cellsPerBlock, groupsPerBlock, sensorsPerBlockv};

		//create matrix for data on GPU and fill it with 0
		std::cout << "\nallocating data area on GPU\n";
		cudaMalloc(&data, groupSizev * groups * sensors * sizeof(Contained));
		cudaMemset(data, 0, groupSizev * groups * sensors * sizeof(Contained));

		//create matrix for zero delay data on GPU and fill it with 0
		std::cout << "\allocating zero delay area on GPU\n";
		cudaMalloc(&zeroDelays, groups * sensors * sizeof(Contained));
		cudaMemset(zeroDelays, 0, groups * sensors * sizeof(Contained));

		//create matrix for accumulator positions for each group on GPU and fill it with 0
		std::cout << "\nallocating accumulators on GPU\n";
		cudaMalloc(&accumulatorsPos, groups * sensors * sizeof(Contained));
		cudaMemset(accumulatorsPos, 0, groups * sensors * sizeof(Contained));

		//create array for info on GPU and fill it
		std::cout << "\nallocating required info on GPU\n";
		cudaMalloc(&info, INFO_AMOUNT * sizeof(std::uint_fast32_t));
		cudaMemcpy(info, tmp, INFO_AMOUNT * sizeof(std::uint_fast32_t), cudaMemcpyHostToDevice);

		//create banks info array on GPU and fill it
		std::cout << "\nallocating optimizations on GPU\n";
		std::vector<std::uint_fast8_t> banksInfoTmp((int)sensors);
		for (int i = 0; i < sensors; ++i) {
			banksInfoTmp[i] = i / sensorsPerBlockv;
		}
		cudaMalloc(&banksInfo, sensors * sizeof(uint_fast8_t));
		cudaMemcpy(banksInfo, banksInfoTmp.data(), sensors * sizeof(std::int_fast8_t), cudaMemcpyHostToDevice);

		std::cout << "\nBinGroupMultiSensor done!\n";
	}




	__host__ ResultArray<Contained> generateResultArray() {
		return ResultArray<Contained>(sensors, groups * groupSizev);
	}



	__host__ std::uint_fast32_t getSensorsPerBlock() {
		return sensorsPerBlockv;
	}


	__host__ std::size_t getTotalSharedMemoryRequired() {
		return sensorsPerBlockv * groupSizev * groups * sizeof(Contained) + sensorsPerBlockv * groups * sizeof(Contained) + +sensorsPerBlockv * groups * sizeof(std::uint_fast8_t) + INFO_AMOUNT * sizeof(std::uint_fast32_t) + sensorsPerBlockv * groupSizev * groups * sizeof(Contained); //last termo is for output memory
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
		std::uint_fast8_t accumulatorPos = getAccumulatorRelativePos(sensor, binGroup, accsPos, accumulatorsPos, info);
		std::uint_fast32_t startOfGroup = getStartOfGroup(sensor, binGroup, info);
		SizeExpModuleMathUnsignedInt pos;
		pos.bitfield = accumulatorPos + 1 + i;
		return data[startOfGroup + pos.bitfield];
	}

	__device__ Contained get(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast32_t i) {
		std::uint_fast8_t accumulatorPos = getAccumulatorRelativePos(sensor, binGroup);
		std::uint_fast32_t startOfGroup = getStartOfGroup(sensor, binGroup);
		SizeExpModuleMathUnsignedInt pos;
		pos.bitfield = accumulatorPos + 1 + i;
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

	__device__ void shift(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		decrementAccumulatorPos(sensor, binGroup);

		//if there is next bin group
		if (binGroup < groupsNum() - 1) {
			addToAccumulator(sensor, binGroup + 1, getAccumulator(sensor, binGroup)); //after the decrement, currentAccumulator is the place where there is the expelled value
			addToZeroDelay(sensor, binGroup + 1, getZeroDelay(sensor, binGroup)); //add current zero delay to the next one
		}

		//clear current
		clearAccumulator(sensor, binGroup);
		clearZeroDelay(sensor, binGroup);
	}




	__device__ Contained getZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* zeroDelays, std::uint_fast32_t* info) {
		return zeroDelays[sensor + binGroup * sensorsPerBlock(info)];
	}

	__device__ Contained getZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		return zeroDelays[groupsPerBlock() * banksInfo[sensor] + binGroup * sensorsPerBlock() + (sensor - sensorsPerBlock() * banksInfo[sensor])];
	}




	__device__ void insertNew(std::uint_fast32_t sensor, Contained datum, Contained* data, Contained* zeroDelays, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		addToAccumulator(sensor, 0, datum, data, accumulatorsPos, info);
		addToZeroDelay(sensor, 0, datum, zeroDelays, info);
	}

	__device__ void insertNew(std::uint_fast32_t sensor, Contained datum) {
		addToAccumulator(sensor, 0, datum);
		addToZeroDelay(sensor, 0);
	}




	__device__ std::uint_fast32_t getAccumulatorRelativePos(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		return accumulatorsPos[sensor + binGroup * sensorsPerBlock(info)];
	}

	__device__ std::uint_fast32_t getAccumulatorRelativePos(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		return accumulatorsPos[groupsPerBlock() * banksInfo[sensor] + binGroup * sensorsPerBlock() + (sensor - sensorsPerBlock() * banksInfo[sensor])];
	}




//======================= INFO GETTERS ===========================================

	__device__ std::uint_fast32_t groupsNum(std::uint_fast32_t* info) {
		return info[1];
	}

	__device__ std::uint_fast32_t groupsNum() {
		return info[1];
	}




	__device__ std::uint_fast32_t sensorsNum(std::uint_fast32_t* info) {
		return info[2];
	}

	__device__ std::uint_fast32_t sensorsNum() {
		return info[2];
	}




	__device__ std::uint_fast32_t groupSize(std::uint_fast32_t* info) {
		return info[0];
	}

	__device__ std::uint_fast32_t groupSize() {
		return info[0];
	}




	__device__ std::uint_fast32_t cellsPerBank(std::uint_fast32_t* info) {
		return info[3];
	}

	__device__ std::uint_fast32_t cellsPerBank() {
		return info[3];
	}




	__device__ std::uint_fast32_t cellsPerBlock(std::uint_fast32_t* info) {
		return info[4];
	}

	__device__ std::uint_fast32_t cellsPerBlock() {
		return info[4];
	}




	__device__ std::uint_fast32_t groupsPerBlock(std::uint_fast32_t* info) {
		return info[5];
	}

	__device__ std::uint_fast32_t groupsPerBlock() {
		return info[5];
	}




	__device__ std::uint_fast32_t sensorsPerBlock(std::uint_fast32_t* info) {
		return info[6];
	}

	__device__ std::uint_fast32_t sensorsPerBlock() {
		return info[6];
	}




	__device__ std::uint_fast32_t getInfo(std::uint_fast8_t i) {
		return info[i];
	}




	private:

	__device__ void decrementAccumulatorPos(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		std::uint_fast32_t accumulatorPos = getAccumulatorRelativePos(sensor, binGroup, accumulatorsPos, info);
		SizeExpModuleMathUnsignedInt newPos;
		newPos.bitfield = accumulatorPos - 1; //does it really work?
		accumulatorsPos[accumulatorPos] = newPos.bitfield;
	}

	__device__ void decrementAccumulatorPos(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		std::uint_fast32_t accumulatorPos = getAccumulatorRelativePos(sensor, binGroup);
		SizeExpModuleMathUnsignedInt newPos;
		newPos.bitfield = accumulatorsPos - 1; //does it really work?
		accumulatorsPos[accumulatorPos] = newPos.bitfield;
	}




	__device__ Contained getAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* data, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		return data[getStartOfGroup(sensor, binGroup, info) + getAccumulatorRelativePos(sensor, binGroup, accumulatorsPos, info)];
	}

	__device__ Contained getAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		return data[getStartOfGroup(sensor, binGroup) + getAccumulatorRelativePos(sensor, binGroup)];
	}




	__device__ void addToAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained add, Contained* data, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		data[getStartOfGroup(sensor, binGroup, info) + getAccumulatorRelativePos(sensor, binGroup, accumulatorsPos, info)] += add;
	}

	__device__ void addToAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained add) {
		data[getStartOfGroup(sensor, binGroup) + getAccumulatorRelativePos(sensor, binGroup)] += add;
	}




	__device__ void clearAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* data, std::uint_fast8_t* accumulatorsPos, std::uint_fast32_t* info) {
		data[getStartOfGroup(sensor, binGroup, info) + getAccumulatorRelativePos(sensor, binGroup, accumulatorsPos, info)] = 0;
	}

	__device__ void clearAccumulator(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		data[getStartOfGroup(sensor, binGroup) + getAccumulatorRelativePos(sensor, binGroup)] = 0;
	}




	__device__ void addToZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained add, Contained* zeroDelays, std::uint_fast32_t* info) {
		zeroDelays[sensor + binGroup * sensorsPerBlock(info)] += add;
	}

	__device__ void addToZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained add) {
		zeroDelays[groupsPerBlock() * banksInfo[sensor] + binGroup * sensorsPerBlock() + (sensor - sensorsPerBlock() * banksInfo[sensor])] += add;
	}




	__device__ void clearZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, Contained* zeroDelays, std::uint_fast32_t* info) {
		zeroDelays[sensor + binGroup * sensorsPerBlock(info)] = 0;
	}

	__device__ void clearZeroDelay(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		zeroDelays[groupsPerBlock() * banksInfo[sensor] + binGroup * sensorsPerBlock() + (sensor - sensorsPerBlock() * banksInfo[sensor])] = 0;
	}




	/**
	 * @brief Returns the first address of the specified group of the specified sensor. The value returned is an address  to the current thread block of execution.
	 * @pre 0 <= sensor < sensorsPerBlock, 0 <= binGroup < binGroupsNum
	*/
	__device__ std::uint_fast32_t getStartOfGroup(std::uint_fast32_t sensor, std::uint_fast32_t binGroup, std::uint_fast32_t* info) {
		return sensor * groupSize(info) + binGroup * cellsPerBank(info);
	}

	__device__ std::uint_fast32_t getStartOfGroup(std::uint_fast32_t sensor, std::uint_fast32_t binGroup) {
		return banksInfo[sensor] * cellsPerBlock() + (sensor - banksInfo[sensor] * sensorsPerBlock()) * groupSize() + binGroup * cellsPerBank();
	}





	//arrays in GPU global memory
	Contained* data;
	Contained* zeroDelays;
	std::uint8_t* accumulatorsPos;
	std::uint_fast32_t* info;
	std::uint_fast8_t* banksInfo;

	//values fo host
	std::uint_fast32_t sensors;
	std::uint_fast32_t groups;
	std::uint_fast32_t groupSizev;
	std::uint_fast32_t sensorsPerBlockv;



	struct SizeExpModuleMathUnsignedInt {
		unsigned int bitfield : SizeExp2;
	};

};

}

#endif
