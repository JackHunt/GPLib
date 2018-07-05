# Set CMake version and project name.
cmake_minimum_required(VERSION 2.8)
project(GPLib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

# Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++17 -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -fPIC -O3 -march=native")
endif(MSVC_IDE)

# Find Eigen.
find_package(Eigen REQUIRED)

# Set the headers and sources.
set(HEADERS
  lib/headers/GaussianProcess.hpp
  lib/headers/GPRegressor.hpp
  lib/headers/Kernels.hpp
  lib/headers/Aliases.hpp
)

set(SOURCES
  lib/GaussianProcess.cpp
  lib/GPRegressor.cpp
  lib/Kernels.cpp
)

# Add library.
add_library(GPLib ${HEADERS} ${SOURCES})

# Add includes.
target_include_directories(GPLib PUBLIC lib/headers/)

# Try to build python bindings.
option(BUILD_PYTHON_BINDINGS "Build Python3 Bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
  #
endif(BUILD_PYTHON_BINDINGS)
