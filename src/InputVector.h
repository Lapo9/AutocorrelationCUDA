#ifndef AUTOCORRELATIONCUDA_INPUTVECTOR
#define AUTOCORRELATIONCUDA_INPUTVECTOR

#include <vector>
#include <string>
#include "CudaInput.h"
#include "DataFile.h"

namespace AutocorrelationCUDA {

/**
* @brief Simple and fast way to provide data for the program.
* @tparam OutType Underlying type of the data stored.
**/
template <typename OutType>
class InputVector final : public CudaInput<OutType> {

	public:

	/**
	* @brief Creates a new InputVector based on the specified vector.
	* @param v The vector to copy.
	**/
	InputVector(std::vector<OutType> v) : base{v} {}

	/**
	* @brief Creates a new InputVector based on the data stored in the specified file.
	* @param path Path to the file.
	* @param fileName Name of the file (without extension).
	* @param format Extension of the file (defaults to .txt).
	**/
	InputVector(const std::string& path, const std::string& fileName, const std::string& format = ".txt") {
		AutocorrelationCUDA::DataFile<OutType> tmp{path, fileName, format};
		base = tmp.read();
	}


	/**
	* @brief Reads all of the vector.
	* @return The underlying vector.
	**/
	std::vector<OutType> read() {
		return std::vector<OutType>{base};
	}


	/**
	* @brief Reads the specified amount of data from the vector, starting where the last read ended.
	* @param valsToRead How many values to read.
	* @return A vector containing the data read.
	**/
	std::vector<OutType> read(uint32 valsToRead) {
		std::vector<OutType> out(valsToRead);

		for (uint32 i = 0; i < valsToRead; ++i) {
			out[i] = base[pos%base.size()];
			pos++;
		}

		return out;
	}

	private:
	std::vector<OutType> base;
	uint32 pos = 0;
};

}

#endif
