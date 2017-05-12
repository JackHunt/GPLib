%module pyGP
%{
#include "lib/typedefs.h"
#include "lib/Kernels.h"
#include "lib/GPRegressor.h"
%}
%ignore GaussianProcess::Kernel;

%include "lib/typedefs.h"
%include "lib/Kernels.h"
%include "lib/GPRegressor.h"
