# Set CMake version and project name.
cmake_minimum_required(VERSION 3.1)
project(GPLib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# Get CPPUtils.
message(CHECK_START "Fetching CPPUtils")
FetchContent_Declare(
  CPPUtils
  GIT_REPOSITORY https://github.com/JackHunt/CPPUtils.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(CPPUtils)
include_directories(${CMAKE_BINARY_DIR}/_deps/cpputils-src/)

# Find Eigen.
message(CHECK_START "Fetching Eigen3")
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(Eigen)
include_directories(${CMAKE_BINARY_DIR}/_deps/eigen-src/)

# Set the headers and sources.
set(GPLIB_HEADERS
  lib/include/GPLib/Aliases.hpp
  lib/include/GPLib/GaussianProcess.hpp
  lib/include/GPLib/GPRegressor.hpp
  lib/include/GPLib/Kernels.hpp
  lib/include/GPLib/Optimisation/Optimiser.hpp
  lib/include/GPLib/Optimisation/LevenbergMarquardt.hpp
  lib/include/GPLib/Optimisation/GradientDescent.hpp
  lib/include/GPLib/Optimisation/ScaledConjugateGradient.hpp
  lib/include/GPLib/Kernels/Kernel.hpp
  lib/include/GPLib/Kernels/SquaredExponential.hpp
)

set(GPLIB_SOURCES
  lib/src/GPRegressor.cpp
  lib/src/Kernels/SquaredExponential.cpp
  lib/src/Optimisation/LevenbergMarquardt.cpp
  lib/src/Optimisation/GradientDescent.cpp
  lib/src/Optimisation/ScaledConjugateGradient.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/lib/include/)

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
