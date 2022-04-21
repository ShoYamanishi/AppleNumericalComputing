#ifndef __SPARSE_MATRIX_VECTOR_METAL_CPP_IMPL_H__
#define __SPARSE_MATRIX_VECTOR_METAL_CPP_IMPL_H__

#include "sparse_matrix_vector_metal_cpp.h"
#include <cstddef>

class SparseMatrixVectorMetalCppImpl
{
  public:
    SparseMatrixVectorMetalCppImpl( const int M, const int N, const int num_nonzero_elems, const bool use_adaptation );

    virtual ~SparseMatrixVectorMetalCppImpl();

    float* getRawPointerOutputVector();

    void setInitialStates(
        int*      csr_block_ptrs,
        int*      csr_threads_per_row,
        int*      csr_max_iters,
        const int num_blocks,
        int*      csr_row_ptrs,
        int*      csr_columns,
        float*    csr_values,
        float*    csr_vector,
        float*    output_vector
    );

    void performComputation();

  private:
    SparseMatrixVectorMetalObjC* m_self;
};

#endif /*__SPARSE_MATRIX_VECTOR_METAL_CPP_IMPL_H__*/
