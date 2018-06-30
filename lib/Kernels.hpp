/*
BSD 3-Clause License

Copyright (c) 2017, Jack Miles Hunt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef GP_KERNELS_HEADER
#define GP_KERNELS_HEADER

#include "typedefs.hpp"

#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace GPLib {
    //Variable name, value.
    typedef std::map<std::string, double> ParameterSet;
	
    //Available kernel types enumerated here.
	enum KernelType{
		SQUARED_EXPONENTIAL
	};
	
    /**
     * @brief The Kernel base class
     */
	class Kernel{
	public:
        /**
         * @brief f Evaluates the kernel value at the given input and hyperParameters.
         * @param a Vector a
         * @param b Vector b
         * @param params Parameter struct.
         * @return Kernel value.
         */
        virtual double f(const Vector &a, const Vector &b, const ParameterSet &params) const = 0;

        /**
         * @brief df Computes a partial derivative of the kernel at the given input and hyperParameters.
         * @param a Vector a
         * @param b Vector b
         * @param params Parameter struct.
         * @param variable Variable to differentiate w.r.t.
         * @return Partial deriivative w.r.t. variable.
         */
        virtual double df(const Vector &a, const Vector &b, const ParameterSet &params, const std::string &variable) const = 0;

        /**
         * @brief ~Kernel
         */
        virtual ~Kernel(){};
	};

    /**
     * @brief The SquaredExponential Kernel class
     */
	class SquaredExponential : public Kernel{
	public:
        /*
         * See base class.
         */
        double f(const Vector &a, const Vector &b, const ParameterSet &params) const;

        /*
         * See base class.
         */
        double df(const Vector &a, const Vector &b, const ParameterSet &params, const std::string &variable) const;
	};
}

#endif
