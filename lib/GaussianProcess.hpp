#ifndef GPLIB_GAUSSIAN_PROCESS_HEADER
#define GPLIB_GAUSSIAN_PROCESS_HEADER

#include <vector>
#include <iostream>

#include "Aliases.hpp"
#include "Kernels.hpp"

namespace GPLib {
    /**
    * @brief jitterChol Performs a cholesky decomposition. Deals with non Pos-Semidef matrices by adding jitter successively to diagonal.
    * @param A Matrix to compute cholesky decomposition of.
    * @param C Matrix to write cholesky decomposition to.
    */
    template<typename T>
    inline void jitterChol(const Matrix<T> &A, Matrix<T> &C) {
        const size_t rowsA = A.rows();
        const size_t colsA = A.cols();
        if (rowsA != colsA) {
            throw std::runtime_error("Cannot take Cholesky Decomposition of non square matrix.");
        }

        Matrix<T> jitter = Matrix<T>::Identity(rowsA, colsA);
        jitter *= 1e-8;

        bool passed = false;

        //Successively add jitter to make positive semi-definite.
        while (!passed && jitter(0, 0) < 1e4) {
            Eigen::LLT< Matrix<T> > chol(A + jitter);
            if (chol.info() == Eigen::NumericalIssue) {//Not pos-semidefinite.
                jitter *= 1.1;
            }
            else {
                passed = true;
                C = chol.matrixL();
            }
        }

        //If Matrix is still not positive semi-definite.
        if (!passed) {
            throw std::runtime_error("Unable to make matrix positive semidefinite.");
        }
    }

    /**
    * @brief buildCovarianceMatrix Builds a Covariance Matrix between two data matrices using a given kernel.
    * @param A Data matrix A.
    * @param B Data matrix B.
    * @param C Output Covariance Matrix.
    * @param params Parameters for kernel(must match kernel choice)
    * @param kernel Kernel(must match Parameters)
    * @param var Variable to differentiate w.r.t if derivative matrix is required.
    */
    template<typename T>
    inline void buildCovarianceMatrix(const Eigen::Map< const Matrix<T> > &A, const Eigen::Map< const Matrix<T> > &B,
        Matrix<T> &C, const ParameterSet<T> &params, const Kernel<T> &kernel,
        const std::string &var = std::string("")) {
        const size_t rowsA = A.rows();
        const size_t rowsB = B.rows();
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < rowsA; i++) {
            for (int j = i + 1; j < rowsB; j++) {
                if (var.compare("") != 0) {
                    C(i, j) = kernel.df(A.row(i), B.row(j), params, var);
                }
                else {
                    C(i, j) = kernel.f(A.row(i), B.row(j), params);
                }
                C(j, i) = C(i, j)
            }
        }
    }

    template<typename T>
    class GaussianProcess {
    protected:
        //Kernel defining this type of regressor.
        Kernel<T> kernel;

    public:
        virtual ~GaussianProcess() {
            //
        }
    };
}

#endif
