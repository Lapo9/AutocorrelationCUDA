#ifndef AUTOCORRELATIONCUDA_DEFINITIONS
#define AUTOCORRELATIONCUDA_DEFINITION


namespace AutocorrelationCUDA {

#define SENSORS 1024
#define SENSORS_PER_BLOCK 8
#define GROUPS_PER_SENSOR 10
#define GROUP_SIZE 32
#define REPETITIONS 20
#define INSTANTS_PER_PACKET 100


#define ACC_POS_START (SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE)
#define ZERO_DELAY_START (ACC_POS_START + SENSORS_PER_BLOCK * GROUPS_PER_SENSOR)
#define SHARED_MEMORY_REQUIRED (SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE + SENSORS_PER_BLOCK * GROUPS_PER_SENSOR + SENSORS_PER_BLOCK * GROUPS_PER_SENSOR + SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE * 4)
#define MAX_LAG (GROUPS_PER_SENSOR * GROUP_SIZE)
#define X32_BITS_PER_BLOCK_ROW SENSORS_PER_BLOCK * GROUP_SIZE / 4 //number of words of 32 bits in each "row" of each block (a row is made up of all of the same bin groups of the sensors of a block)
#define X32_BITS_PER_BLOCK_DATA SENSORS_PER_BLOCK * GROUPS_PER_SENSOR * GROUP_SIZE / 4 //number of words of 32 bits in each block (only cells in the data array)
#define X32_BITS_PER_BLOCK_ZD_ACC SENSORS_PER_BLOCK * GROUPS_PER_SENSOR / 4 //number of words of 32 bits in each block (only cells in the zero delay or accumulators pos array)

using uint8 = std::uint_fast8_t;
using uint16 = std::uint_fast16_t;
using uint32 = std::uint_fast32_t;

}


#endif 

