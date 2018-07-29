#include "bindings.hpp"

using namespace GPLib;
using namespace GPLib::Kernels;

PYBIND11_MODULE(MODULE_NAME, m) {
    // Bind Squared Exponential Kernel.
    pybind11::class_< GPRegressor <DTYPE> >(m, "GPRegressor")
        .def(pybind11::init< Kernel<DTYPE> >()
}