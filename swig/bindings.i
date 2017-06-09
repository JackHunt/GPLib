%module pyGP
%{
#define SWIG_FILE_WITH_INIT
#include "lib/typedefs.h"
#include "lib/Kernels.h"
#include "lib/GPRegressor.h"
#include "lib/GDOptimiser.h"
%}

%include "numpy.i"
%include "std_string.i"
%include "std_map.i"
%include "std_vector.i"
%include "std_shared_ptr.i"

%template(MapStringDouble) std::map<std::string, double>;
%template(DoubleVector) std::vector<double>;
%shared_ptr(GaussianProcess::GPRegressor)

%init %{
  import_array();
%}

%apply (double *IN_ARRAY2, int DIM1, int DIM2) {(const double *trainData, int trainCols, int trainRows)};
%apply (double *IN_ARRAY1, int DIM1) {(const double *trainTruth, int trainTruthRows)};
%apply (double *IN_ARRAY2, int DIM1, int DIM2) {(const double *testData, int testCols, int testRows)};
%apply (double *IN_ARRAY1, int DIM1) {(const double *testTruth, int testTruthRows)};

%include "lib/typedefs.h"
%include "lib/Kernels.h"
%include "lib/GPRegressor.h"
%include "lib/GDOptimiser.h"
