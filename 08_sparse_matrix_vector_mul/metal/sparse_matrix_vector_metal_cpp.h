#ifndef __SPARSE_MATRIX_VECTOR_METAL_CPP_H__
#define __SPARSE_MATRIX_VECTOR_METAL_CPP_H__

#include <simd/simd.h>
typedef unsigned int uint;

struct sparse_matrix_vector_constants {
    uint  num_rows;
    uint  num_columns;
    uint  num_nnz;
    uint  num_blocks;
};

class SparseMatrixVectorMetalCppImpl;

class SparseMatrixVectorMetalCpp
{

  public:
    SparseMatrixVectorMetalCpp( const int M, const int N, const int num_nonzero_elems, const bool use_adaptation );

    virtual ~SparseMatrixVectorMetalCpp();

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
    SparseMatrixVectorMetalCppImpl* m_impl;

};

#endif /*__SPARSE_MATRIX_VECTOR_METAL_CPP_H__*/
