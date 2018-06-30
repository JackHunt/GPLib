#Set CMake version and project name.
cmake_minimum_required(VERSION 2.8)
project(GPLib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++11 -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -fPIC -O3 -march=native")
endif(MSVC_IDE)

#Determine if we can use OpenMP and append to compiler flags accordingly.
option(USE_OpenMP "Use OpenMP" ON)
if(USE_OpenMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	add_definitions(-DWITH_OPENMP)
  endif(OPENMP_FOUND)
endif(USE_OpenMP)

#Find Eigen.
find_package(Eigen REQUIRED)

#Set the headers and sources.
set(HEADERS
  lib/GPRegressor.hpp
  lib/GDOptimiser.hpp
  lib/Kernels.hpp
  lib/Util.hpp
  lib/Aliases.hpp
)

set(SOURCES
  lib/GPRegressor.cpp
  lib/GDOptimiser.cpp
  lib/Kernels.cpp
)

#Add library.
add_library(GPLib ${HEADERS} ${SOURCES})

#Try and find SWIG to build python bindings.
option(BUILD_PYTHON_BINDINGS "Build Python3 Bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
  #
endif(BUILD_PYTHON_BINDINGS)
