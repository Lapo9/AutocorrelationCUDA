#ifndef AUTOCORRELATIONCUDA_CUDAINPUT
#define AUTOCORRELATIONCUDA_CUDAINPUT

#include <vector>

namespace AutocorrelationCUDA {

/**
* @brief Abstraction implemented by the objects responsible for providing the data from the sensors to the program.
**/
template <typename OutType>
class CudaInput {
	public:
	/**
	* @brief Reads all available data.
	**/
	virtual std::vector<OutType> read() = 0;

	/**
	* @brief Reads the specified amount of data.
	**/
	virtual std::vector<OutType> read(std::uint_fast32_t valsToRead) = 0;
};
}


#endif

