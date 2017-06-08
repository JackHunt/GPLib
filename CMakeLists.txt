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

#Try and find SWIG to build python bindings.
option(BUILD_PYTHON_BINDINGS "Build Python3 Bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
  find_package(SWIG)
  if(SWIG_FOUND)
	add_definitions(-DWITH_PYTHON_BINDINGS)
    include(${SWIG_USE_FILE})
    find_package(PythonLibs 3)
	include_directories(${PYTHON_INCLUDE_PATH})
	include_directories(${CMAKE_CURRENT_SOURCE_DIR})
	set(CMAKE_SWIG_FLAGS "")
	set_source_files_properties(swig/bindings.i PROPERTIES CPLUSPLUS ON)
#	set_source_files_properties(swig/bindings.i PROPERTIES SWIG_FLAGS "-includeall")
	swig_add_module(pyGP python swig/bindings.i ${SOURCES})
	swig_link_libraries(pyGP ${PYTHON_LIBRARIES} ${GaussianProcess})
	target_link_libraries(_pyGP ${GaussianProcess})
  endif(SWIG_FOUND)
endif(BUILD_PYTHON_BINDINGS)
