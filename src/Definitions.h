#ifndef AUTOCORRELATIONCUDA_DEFINITIONS
#define AUTOCORRELATIONCUDA_DEFINITIONS

/**
* @brief Contains all of the classes/definitions used for this project.
**/
namespace AutocorrelationCUDA {

using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;

/**
* @brief Number of sensors.
**/
constexpr uint16 SENSORS = 1024;

/**
* @brief Number of sensors in each CUDA block.
**/
constexpr uint8 SENSORS_PER_BLOCK = 8;

/**
* @brief Number of bin groups for each sensor.
**/
constexpr uint8 GROUPS_PER_SENSOR = 10;

/**
* @brief Number of elements in each group.
**/
constexpr uint8 GROUP_SIZE = 32;

/**
* @brief How many times to run the algorithm.
**/
constexpr uint16 REPETITIONS = 1;

/**
* @brief Number of values in each packet sent to the GPU.
**/
constexpr uint32 INSTANTS_PER_PACKET = 400;



constexpr uint16 ACC_POS_START = SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE;

constexpr uint16 ZERO_DELAY_START = ACC_POS_START + SENSORS_PER_BLOCK * GROUPS_PER_SENSOR;

constexpr uint16 ELEMS_REQUIRED_FOR_BIN_STRUCTURE = SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE + SENSORS_PER_BLOCK * GROUPS_PER_SENSOR + SENSORS_PER_BLOCK * GROUPS_PER_SENSOR;

constexpr uint16 ELEMS_REQUIRED_FOR_OUTPUT = SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE;

constexpr uint16 MAX_LAG = GROUPS_PER_SENSOR * GROUP_SIZE;

constexpr uint16 X32_BITS_PER_BLOCK_ROW = SENSORS_PER_BLOCK * GROUP_SIZE / 2; //number of words of 32 bits in each "row" of each block (a row is made up of all of the same bin groups of the sensors of a block)

constexpr uint16 X32_BITS_PER_BLOCK_DATA = SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE / 2; //number of words of 32 bits in each block (only cells in the data array)

constexpr uint16 X32_BITS_PER_BLOCK_ZD_ACC = SENSORS_PER_BLOCK * GROUPS_PER_SENSOR / 2; //number of words of 32 bits in each block (only cells in the zero delay or accumulators pos array)

constexpr uint16 COPY_REPETITIONS = (GROUPS_PER_SENSOR % 2) == 0 ? GROUPS_PER_SENSOR / 2 : GROUPS_PER_SENSOR / 2 + 1;



}


#endif 

