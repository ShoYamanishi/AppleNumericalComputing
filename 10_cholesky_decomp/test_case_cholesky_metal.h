#ifndef __TEST_CASE_CHOLESKY_METAL_H__
#define __TEST_CASE_CHOLESKY_METAL_H__

#include "cholesky_metal_cpp.h"

#include "test_case_cholesky.h"

template<class T, bool IS_COL_MAJOR>
class TestCaseCholesky_metal : public TestCaseCholesky<T, IS_COL_MAJOR> {

    const bool       m_use_mps;
    CholeskyMetalCpp m_metal;

  public:

    TestCaseCholesky_metal( const int dim , const bool use_mps )
        :TestCaseCholesky<T, IS_COL_MAJOR>( dim )
        ,m_use_mps(use_mps)
        ,m_metal( dim, use_mps )
    {
        static_assert( std::is_same< float,T >::value );
        static_assert( IS_COL_MAJOR );

        this->setMetal( use_mps ? MPS : DEFAULT, 1, 1 );
    }

    virtual ~TestCaseCholesky_metal(){ ; }

    virtual void setInitialStates( T* L, T* b ) {

        m_metal.setInitialStates( L, b );
        TestCaseCholesky<T,IS_COL_MAJOR>::setInitialStates( L, b );
    }

    virtual void compareTruth( const T* const L_baseline, const T* const x_baseline ) {

        float* p =  m_metal.getRawPointerL();

        if ( m_use_mps ) {

            for (int i = 0 ; i < this->m_dim ; i++ ) {
                for (int j = 0 ; j <= i ; j++ ) {
                    this->m_L[ lower_mat_index<IS_COL_MAJOR>( i, j, this->m_dim ) ] = p[ this->m_dim * i + j ];
                }
            }
        }
        else {
            memcpy( this->m_L, m_metal.getRawPointerL(), (this->m_dim + 1) * this->m_dim * sizeof(float) / 2 );
        }
        memcpy( this->m_x, m_metal.getRawPointerX(), this->m_dim * sizeof(float) );

        TestCaseCholesky<T,IS_COL_MAJOR>::compareTruth( L_baseline, x_baseline );
    }

    void run() {
        m_metal.performComputation();
    }
};

#endif /*__TEST_CASE_CHOLESKY_METAL_H__*/
