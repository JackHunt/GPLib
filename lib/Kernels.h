#ifndef GP_KERNELS_HEADER
#define GP_KERNELS_HEADER

#include "typedefs.h"

#include <string>
#include <map>
#include <cmath>
#include <Eigen/Dense>

namespace GaussianProcess{
	typedef std::map<std::string, double> ParamaterSet;
	
	enum KernelType{
		SQUARED_EXPONENTIAL
	};
	
	class Kernel{
	public:
		virtual double f(const Vector &a, const Vector &b, const ParamaterSet &params) const = 0;
		virtual double df(const Vector &a, const Vector &b, const std::string &var, const ParamaterSet &params) const = 0;
		virtual ~Kernel(){};
	};

	class SquaredExponential : public Kernel{
	public:
		double f(const Vector &a, const Vector &b, const ParamaterSet &params) const;
		double df(const Vector &a, const Vector &b, const std::string &var, const ParamaterSet &params) const;
	};
}

#endif
