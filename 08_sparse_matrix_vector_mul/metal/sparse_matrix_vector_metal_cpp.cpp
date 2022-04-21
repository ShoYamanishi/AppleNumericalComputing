class SparseMatrixVectorMetalObjC;
#include "sparse_matrix_vector_metal_cpp_impl.h"
#include "sparse_matrix_vector_metal_cpp.h"

SparseMatrixVectorMetalCpp::SparseMatrixVectorMetalCpp(
    const int M,
    const int N,
    const int num_nonzero_elems,
    const bool use_adaptation
)
    :m_impl( new SparseMatrixVectorMetalCppImpl( M, N, num_nonzero_elems, use_adaptation ) )
{
    ;
}

SparseMatrixVectorMetalCpp::~SparseMatrixVectorMetalCpp()
{
    delete m_impl;
}

float* SparseMatrixVectorMetalCpp::getRawPointerOutputVector()
{
    return m_impl->getRawPointerOutputVector();
}

void SparseMatrixVectorMetalCpp::setInitialStates(
    int*      csr_block_ptrs,
    int*      csr_threads_per_row,
    int*      csr_max_iters,
    const int num_blocks,
    int*      csr_row_ptrs,
    int*      csr_columns,
    float*    csr_values,
    float*    csr_vector,
    float*    output_vector
) {
    return m_impl->setInitialStates(
        csr_block_ptrs,
        csr_threads_per_row,
        csr_max_iters, 
        num_blocks,
        csr_row_ptrs,
        csr_columns,
        csr_values,
        csr_vector,
        output_vector
    );
}

void SparseMatrixVectorMetalCpp::performComputation()
{
    m_impl->performComputation();
}
