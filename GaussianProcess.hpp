#ifndef GPLIB_GAUSSIAN_PROCESS_HEADER
#define GPLIB_GAUSSIAN_PROCESS_HEADER

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <iostream>

#include "Kernels.hpp"
#include "Aliases.hpp"
#include "Util.hpp"

namespace GPLib {
    class GaussianProcess {
    protected:
        //Kernel defining this type of regressor.
        Kernel<T> kernel;

    public:

    };
}

#endif
