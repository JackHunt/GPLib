# Set CMake version and project name.
cmake_minimum_required(VERSION 2.8)
project(GPLib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

# Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /D_USE_MATH_DEFINES")
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++17 -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -fPIC -O3 -march=native")
endif(MSVC_IDE)

# Find Eigen.
find_package(Eigen REQUIRED)

# Set the headers and sources.
set(GPLIB_HEADERS
  lib/headers/Aliases.hpp
  lib/headers/GaussianProcess.hpp
  lib/headers/GPRegressor.hpp
  lib/headers/Kernels.hpp
  lib/headers/Optimisation/Optimiser.hpp
  lib/headers/Kernels/Kernel.hpp
  lib/headers/Kernels/SquaredExponential.hpp
)

set(GPLIB_SOURCES
  lib/GPRegressor.cpp
  lib/Kernels/SquaredExponential.cpp
)

# Add library.
add_library(GPLib STATIC ${GPLIB_HEADERS} ${GPLIB_SOURCES})

# Add includes.
target_include_directories(GPLib PUBLIC lib/headers/)

# Add demo executable.
add_executable(Demo demo/demo.hpp demo/demo.cpp)
# C:/Users/Jack\ Hunt/GitRepos/GPLib/build/Debug/GPLib.lib
target_link_libraries(Demo GPLib)

# Add includes.
target_include_directories(Demo PUBLIC lib/headers/)
