#Set CMake version and project name.
cmake_minimum_required(VERSION 2.8)
project(GaussianProcess)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++11 -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -fPIC -O3 -march=native")
endif(MSVC_IDE)

#Find Eigen.
find_package(Eigen REQUIRED)

#Set the headers and sources.
set(HEADERS
  lib/GPRegressor.h
  lib/LMOptimiser.h
  lib/Kernels.h
)

set(SOURCES
  lib/GPRegressor.cpp
  lib/LMOptimiser.cpp
  lib/Kernels.cpp
)

#Add library.
add_library(GaussianProcess ${HEADERS} ${SOURCES})
