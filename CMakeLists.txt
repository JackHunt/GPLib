# Set CMake version and project name.
cmake_minimum_required(VERSION 3.1)
project(GPLib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

# Find Eigen.
find_package(Eigen REQUIRED)

# Set the headers and sources.
set(GPLIB_HEADERS
  lib/headers/Aliases.hpp
  lib/headers/GaussianProcess.hpp
  lib/headers/GPRegressor.hpp
  lib/headers/Kernels.hpp
  lib/headers/Optimisation/Optimiser.hpp
  lib/headers/Optimisation/LevenbergMarquardt.hpp
  lib/headers/Optimisation/GradientDescent.hpp
  lib/headers/Optimisation/ScaledConjugateGradient.hpp
  lib/headers/Kernels/Kernel.hpp
  lib/headers/Kernels/SquaredExponential.hpp
)

set(GPLIB_SOURCES
  lib/GPRegressor.cpp
  lib/Kernels/SquaredExponential.cpp
  lib/Optimisation/LevenbergMarquardt.cpp
  lib/Optimisation/GradientDescent.cpp
  lib/Optimisation/ScaledConjugateGradient.cpp
)

# Add library.
add_library(GPLib STATIC ${GPLIB_HEADERS} ${GPLIB_SOURCES})
set_target_properties(GPLib PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

# Add includes.
target_include_directories(GPLib PUBLIC lib/headers/)
target_include_directories(GPLib PUBLIC lib/headers/CPPUtils)

# Add demo executable.
add_executable(Demo demo/demo.hpp demo/demo.cpp)

target_link_libraries(Demo GPLib)

# Add includes.
target_include_directories(Demo PUBLIC lib/headers/)
