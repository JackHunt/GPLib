#ifndef GPLIB_GAUSSIAN_PROCESS_HEADER
#define GPLIB_GAUSSIAN_PROCESS_HEADER

#include <vector>
#include <iostream>
#include <memory>

#include "Aliases.hpp"
#include "Kernels.hpp"

namespace GPLib {
    template<typename T>
    class GaussianProcess {
    protected:
        static void jitterChol(const Matrix<T> &A, Matrix<T> &C);
        
        static void buildCovarianceMatrix(const MapMatrix<T> &A, const MapMatrix<T> &B,
                                          Matrix<T> &C, const ParameterSet<T> &params,
                                          const std::shared_ptr< Kernel<T> > &kernel, 
                                          const std::string &var = std::string(""));

        virtual T logLikelihood() = 0;

        virtual Vector<T> logLikelihoodGrad() = 0;

    protected:
        // Covariance Kernel defining this type of regressor.
        std::shared_ptr< Kernel<T> > kernel;

        //Noise to be added to kernel diagonal.
        T jitter = 1.0;

    public:
        GaussianProcess(KernelType kernType);

        virtual ~GaussianProcess();

        Kernel<T> getKernel() const;

        virtual void train() = 0;

        virtual void predict() const = 0;
    };
}

#endif
