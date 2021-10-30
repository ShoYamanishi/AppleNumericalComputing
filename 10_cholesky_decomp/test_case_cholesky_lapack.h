#ifndef __TEST_CASE_CHOLESKY_LAPACK_H__
#define __TEST_CASE_CHOLESKY_LAPACK_H__

#include <Accelerate/Accelerate.h>

#include "test_case_cholesky.h"

template<class T, bool IS_COL_MAJOR>
class TestCaseCholesky_lapack : public TestCaseCholesky<T, IS_COL_MAJOR> {

    T* m_lapA;
    T* m_lapbx;

  public:

    TestCaseCholesky_lapack( const int dim )
        :TestCaseCholesky<T, IS_COL_MAJOR>( dim )
        ,m_lapA ( new T[ dim * dim ] )
        ,m_lapbx( new T[ dim ] )
    {
        static_assert( std::is_same< float,T >::value || std::is_same< double,T >::value );

        this->setImplementationType( LAPACK );
    }

    virtual ~TestCaseCholesky_lapack() {

        delete[] m_lapA;
        delete[] m_lapbx;
    }

    virtual void setInitialStates( T* A, T* b ) {

        for ( int i = 0; i < this->m_dim; i++ ) {

            for ( int j = 0; j <= i; j++ ) {

                const T val = A[ lower_mat_index<IS_COL_MAJOR>( i, j, this->m_dim ) ];

                m_lapA[ i * this->m_dim + j ] = val;
                if ( i != j ) {
                    m_lapA[ j * this->m_dim + i ] = val;
                }
            }
            m_lapbx[ i ] = b[i];
        }

        TestCaseCholesky<T,IS_COL_MAJOR>::setInitialStates( A, b );
    }

    virtual void compareTruth( const T* const L_baseline, const T* const x_baseline ) {

        for ( int i = 0; i < this->m_dim; i++ ) {

            this->m_x[i] = m_lapbx[i];

            for ( int j = 0; j <= i; j++ ) {

                this->m_L[ lower_mat_index<IS_COL_MAJOR>( i, j, this->m_dim ) ] = m_lapA[ i + (this->m_dim * j) ];
            }
        }

        TestCaseCholesky<T,IS_COL_MAJOR>::compareTruth( L_baseline, x_baseline );
    }

    void run() {

        char   uplo[2] = "L";
        int    n       = this->m_dim;
        int    nrhs    = 1;
        int    lda     = this->m_dim;
        int    ldb     = this->m_dim;
        int    info;
        int    r;
        if constexpr ( std::is_same< float,T >::value ) {
            r = sposv_( uplo, &n, &nrhs, m_lapA, &lda, m_lapbx, &ldb, &info );
        }
        else {
            r = dposv_( uplo, &n, &nrhs, m_lapA, &lda, m_lapbx, &ldb, &info );
        }
        if ( r != 0 ) {
            std::cerr << "sposv returned non zeror:" << r << " info:"  << info << "\n";
        }

    }
};

#endif /*__TEST_CASE_CHOLESKY_LAPACK_H__*/
