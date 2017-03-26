#ifndef GP_KERNELS_HEADER
#define GP_KERNELS_HEADER

#include <string>
#include <map>
#include <cmath>
#include <Eigen/Dense>

namespace GaussianProcess{
	typedef std::map<std::string, float> ParamaterSet;
	
	enum KernelType{
		SQUARED_EXPONENTIAL
	};
	
	class Kernel{
	public:
		virtual float f(const Eigen::VectorXf &a, const Eigen::VectorXf &b, const ParamaterSet &params) const = 0;
		virtual float df(const Eigen::VectorXf &a, const Eigen::VectorXf &b, const std::string &var, const ParamaterSet &params) const = 0;
		virtual ~Kernel();
	};

	class SquaredExponential : public Kernel{
	public:
		float f(const Eigen::VectorXf &a, const Eigen::VectorXf &b, const ParamaterSet &params) const;
		float df(const Eigen::VectorXf &a, const Eigen::VectorXf &b, const std::string &var, const ParamaterSet &params) const;
	};
}

#endif
