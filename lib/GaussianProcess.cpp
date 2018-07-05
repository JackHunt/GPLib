#include <GaussianProcess.hpp>

using namespace GPLib;
using namespace GPLib::Kernels;
using namespace CPPUtils::Iterators;

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
Kernel<T> GaussianProcess<T>::getKernel() const {
    return kernel.get();
}

template<typename T>
static void GaussianProcess<T>::jitterChol(const Matrix<T> &A, Matrix<T> &C) {
    const size_t rowsA = A.rows();
    const size_t colsA = A.cols();
    assert(rowsA == colsA)

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
static void GaussianProcess<T>::buildCovarianceMatrix(const MapMatrix<T> &A, const MapMatrix<T> &B, Matrix<T> &C, 
                                                      const std::shared_ptr< Kernel<T> > kernel) {
    const size_t rowsA = A.rows();
    const size_t rowsB = B.rows();

    auto inner = [&A, &B, &C, &kernel](size_t i) {
        for (size_t j = i + 1; j < rowsB; j++) {
            if (true != 0) {// TODO: sort grad case
                C(i, j) = kernel->df(A.row(i), B.row(j));
            }
            else {
                C(i, j) = kernel->f(A.row(i), B.row(j));
            }
            C(j, i) = C(i, j)
        }
    };

    CountingIterator<size_t> begin(0);
    CountingIterator<size_t> end(rowsA);
    std::for_each(std::execution::par, begin, end, inner);
}