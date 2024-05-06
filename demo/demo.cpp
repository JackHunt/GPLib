#include <string>
#include <iostream>

#include <CPPUtils/IO/CSVFile.hpp>

#include <GPLib/GPRegressor.hpp>

using CPPUtils::IO::CSVFile;

using namespace GPLib;

int main(int argc, char *argv[]) {
    // Verify a file is provided.
    if (argc < 2) {
        std::cout << "USAGE: ./demo <in_file>" << std::endl;
        return 0;
    }

    // Load the CSV file.
    CSVFile<double, int> data;
    data.readFromDisk(std::string(argv[1]));
    std::cout << argv[1] << std::endl;
    std::cout << data.getNumRows() << std::endl;

    GPRegressor<double> regressor;
}
