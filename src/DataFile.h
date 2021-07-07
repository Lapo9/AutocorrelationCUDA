#ifndef AUTOCORRELATIONCUDA_DATAFILE
#define AUTOCORRELATIONCUDA_DATAFILE

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "CudaInput.h"
#include "Timer.h"


namespace AutocorrelationCUDA {
	
//it is common to pass to this class a file that contains uint8_t, but the common >> operator would read them as char (for example 2 -> '2' = 55). We have to read 2 as 2 (numerical value).
std::istream& operator>>(std::istream& file, std::uint8_t& val) {
		std::uint_fast32_t numVal;
		file >> numVal;
		val = numVal;
		return file;
	}



/** 
* @brief Simple helper to read the file where test data is stored.
* @tparam ConatinedType Type of data contained by the file.
**/
template<typename ContainedType>
class DataFile final : public CudaInput<ContainedType> {
	
	public:

	/**
	* @brief Creats a new DataFile object, based on the specified parameters.
	* @param path Path to the file.
	* @param fileName Name of the file (without extension).
	* @param format Extension of the file (defaults to .txt).
	**/
	DataFile(const std::string& path, const std::string& fileName, const std::string& format = ".txt") : file{path+fileName+format}, fileName{fileName}, format{format} {
		if (!file.is_open()) {
			throw std::runtime_error("Cannot open the file: " + path + fileName + format);
		}
	}

	/**
	* @brief Reads the specified amount of values and return them in a vector. If valsToRead is less then the remaining length of the file, read till eof.
	* @param valsToRead How many values to read from the file.
	* @return Vector containing the data read.
	**/
	std::vector<ContainedType> read(std::uint_fast32_t valsToRead){
		std::vector<ContainedType> vals{};
		ContainedType tmp;

		for(std::uint_fast32_t i = 0; i < valsToRead && file >> tmp; ++i) {
			vals.push_back(tmp);
		}

		return vals;
	}


	/**
	* @brief Reads all of the values in the file.
	* @return Vector containing the data read.
	**/
	std::vector<ContainedType> read() {
		std::vector<ContainedType> vals{};
		ContainedType tmp;

		while(file >> tmp) {
			vals.push_back(tmp);
		}

		return vals;
	}


	/**
	* @brief Writes the vector to a file. The file is saved in the current directory.
	* @tparam OutType Type of data to be stored to the file. Defaults to the input file data type.
	* @param data Data to save to the external file.
	* @param name Relative path from current directory and name of the file. Defaults to "out_data.txt".
	* @param separator How to divide the written data in the file. Defaults to "\n".
	**/
	template <typename OutType = ContainedType>
	static void write(const std::vector<OutType>& data, const std::string& name = "out_data.txt", const std::string& separator = "\n"){
		std::fstream out{name, std::ios_base::out};
		for (OutType val : data) {
			out << val << separator;
		}
	}



	private:

	std::ifstream file;
	std::string fileName;
	std::string format;


};




}



#endif

