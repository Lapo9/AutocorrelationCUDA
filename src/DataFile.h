#ifndef AUTOCORRELATIONCUDA_DATAFILE
#define AUTOCORRELATIONCUDA_DATAFILE

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "CudaInput.h"


namespace AutocorrelationCUDA {
	
//it is common to pass to this class a file that contains uint8_t, but the common >> operator would read them as char (for example 2 -> '2' = 55). We have to read 2 as 2 (numerical value).
std::istream& operator>>(std::istream& file, std::uint8_t& val) {
		int numVal;
		file >> numVal;
		val = numVal;
		return file;
	}



//this class is a simple helper to read the file where test data is stored
template<typename ContainedType>
class DataFile final : public CudaInput<ContainedType> {
	
	public:

	DataFile(const std::string& path, const std::string& fileName, const std::string& format = ".txt") : file{path+fileName+format}, fileName{fileName}, format{format} {
		if (!file.is_open()) {
			throw std::runtime_error("Cannot open the file: " + path + fileName + format);
		}
	}

	//Read valsToRead values and return them in an array. If valsToRead is less then the remaining length of the file, read till eof.
	std::vector<ContainedType> read(unsigned int valsToRead){
		std::vector<ContainedType> vals{};
		ContainedType tmp;

		for(int i = 0; i < valsToRead && file >> tmp; ++i) {
			vals.push_back(tmp);
		}

		return vals;
	}


	//Read all the values in the file and return them in an array.
	std::vector<ContainedType> read() {
		std::vector<ContainedType> vals{};
		ContainedType tmp;

		while(file >> tmp) {
			vals.push_back(tmp);
		}

		return vals;
	}


	//Write the vector to a file named as the input file with suffix "_out". 
	//The file is saved in the current directory.
	template <typename OutType = ContainedType>
	void write(const std::vector<OutType>& data, const std::string& separator = "\n"){
		std::fstream out{fileName + "_out" + format, std::ios_base::out};
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

