#include "GaussianProcess.hpp"

using namespace GPLib;

template<typename T>
GaussianProcess<T>::GaussianProcess(KernelType kernType) {
    switch (kernType) {
    case KernelType::SQUARED_EXPONENTIAL:
        kernel.reset(new SquaredExponential<T>());
        break;
    default:
        throw std::runtime_error("Invalid kernel choice.");
    }
}

template<typename T>
GaussianProcess<T>::~GaussianProcess() {
    //
}

template<typename T>
void GaussianProcess<T>::jitterChol(const Matrix<T> &A, Matrix<T> &C) {
    const size_t rowsA = A.rows();
    const size_t colsA = A.cols();
    if (rowsA != colsA) {
        throw std::runtime_error("Cannot take Cholesky Decomposition of non square matrix.");
    }

    Matrix<T> jitter = Matrix<T>::Identity(rowsA, colsA);
    jitter *= 1e-8;

    bool passed = false;

    // Successively add jitter to make positive semi-definite.
    while (!passed && jitter(0, 0) < 1e4) {
        Eigen::LLT< Matrix<T> > chol(A + jitter);
        if (chol.info() == Eigen::NumericalIssue) {// Not pos-semidefinite.
            jitter *= 1.1;
        }
        else {
            passed = true;
            C = chol.matrixL();
        }
    }

    // If Matrix is still not positive semi-definite.
    if (!passed) {
        throw std::runtime_error("Unable to make matrix positive semidefinite.");
    }
}

template<typename T>
void GaussianProcess<T>::buildCovarianceMatrix(const MapMatrix<T> &A, const MapMatrix<T> &B,
                                               Matrix<T> &C, const ParameterSet<T> &params, 
                                               const std::shared_ptr< Kernel<T> > &kernel, 
                                               const std::string &var) {
    const size_t rowsA = A.rows();
    const size_t rowsB = B.rows();

#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < rowsA; i++) {
        for (int j = i + 1; j < rowsB; j++) {
            if (var.compare("") != 0) {
                C(i, j) = kernel->df(A.row(i), B.row(j), params, var);
            }
            else {
                C(i, j) = kernel->f(A.row(i), B.row(j), params);
            }
            C(j, i) = C(i, j)
        }
    }
}