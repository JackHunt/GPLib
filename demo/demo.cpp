#include "demo.hpp"

using namespace GPLib;

int main(int argc, char *argv[]) {
    // Verify a file is provided.
    if (argc < 1) {
        std::cout << "USAGE: ./demo <in_file>" << std::endl;
        return 0;
    }

    // Load the CSV file.
    CSVFile<double, int> data;
    data.readFromDisk(std::string(argv[1]));
    std::cout << data.getNumRows() << std::endl;

    GPRegressor<double> regressor;
}