#ifndef __TEST_CASE_CHOLESKY_LAPACK_INVERSE_H__
#define __TEST_CASE_CHOLESKY_LAPACK_INVERSE_H__

#include <Accelerate/Accelerate.h>

#include "test_case_cholesky.h"

template<class T, bool IS_COL_MAJOR>
class TestCaseCholesky_lapack_inverse : public TestCaseCholesky<T, IS_COL_MAJOR> {

    T* m_lapA;
    T* m_lapAinv;
    T* m_lapAorg;
    T* m_lapAAinv;

  public:

    TestCaseCholesky_lapack_inverse( const int dim )
        :TestCaseCholesky<T, IS_COL_MAJOR>( dim )
        ,m_lapA ( new T[ dim * dim ] )
        ,m_lapAinv ( new T[ dim * (dim+1) ] )
        ,m_lapAorg ( new T[ dim * dim ] )
        ,m_lapAAinv ( new T[ dim * dim ] )
    {
        static_assert( std::is_same< float,T >::value || std::is_same< double,T >::value );

        this->setImplementationType( LAPACK_WITH_MAT_INVERSE );
    }

    virtual ~TestCaseCholesky_lapack_inverse() {

        delete[] m_lapA;
        delete[] m_lapAinv;
        delete[] m_lapAorg;
        delete[] m_lapAAinv;
    }

    virtual void setInitialStates( T* A, T* b ) {

        for ( int i = 0; i < this->m_dim; i++ ) {

            for ( int j = 0; j <= i; j++ ) {

                const T val = A[ lower_mat_index<IS_COL_MAJOR>( i, j, this->m_dim ) ];

                m_lapA[ i * this->m_dim + j ] = val;
                m_lapAorg[ i * this->m_dim + j ] = val;
                if ( i != j ) {
                    m_lapA[ j * this->m_dim + i ] = val;
                    m_lapAorg[ j * this->m_dim + i ] = val;
                }

                if ( i == j ){
                    m_lapAinv[ j * this->m_dim + i ] = 1.0;
                }
                else {
                    m_lapAinv[ j * this->m_dim + i ] = 0.0;
                    m_lapAinv[ i * this->m_dim + j ] = 0.0;
                }
            }
            m_lapAinv[ this->m_dim * this->m_dim + i ] = b[i];
        }

        TestCaseCholesky<T,IS_COL_MAJOR>::setInitialStates( A, b );
    }

    virtual void compareTruth( const T* const L_baseline, const T* const x_baseline ) {

        for ( int i = 0; i < this->m_dim; i++ ) {

            this->m_x[i] = m_lapAinv[ this->m_dim * this->m_dim + i ];

            for ( int j = 0; j <= i; j++ ) {

                this->m_L[ lower_mat_index<IS_COL_MAJOR>( i, j, this->m_dim ) ] = m_lapA[ i + (this->m_dim * j) ];
            }
        }

         T total_error = 0.0;

        for ( int i = 0; i < this->m_dim; i++ ) {

            T row_error = 0.0;

            for ( int j = 0; j < this->m_dim; j++ ) {

                T output = 0.0;

                for ( int k = 0; k < this->m_dim; k++ ) {

                    const T a_ik = m_lapAorg[ k * this->m_dim + i ];

                    const T ainv_kj = m_lapAinv[ j * this->m_dim + k ];

                    output += (a_ik * ainv_kj);
                }
                if (i==j) {
                    row_error += fabs(output - 1.0);
                    total_error += fabs(output - 1.0);
                }
                else {
                    row_error += fabs(output);
                    total_error += fabs(output);
                }
            }

            // add the error on A*Ainv to m_x;
            this->m_x[i] += ( row_error / (T)(this->m_dim) );
        }

        cerr << "Total error on A*Ainv with dim [" << this->m_dim << "]: " << total_error << "\n";

        TestCaseCholesky<T,IS_COL_MAJOR>::compareTruth( L_baseline, x_baseline );
    }

    void run() {

        char   uplo[2] = "L";
        int    n       = this->m_dim;
        int    nrhs    = this->m_dim + 1;
        int    lda     = this->m_dim;
        int    ldb     = this->m_dim;
        int    info;
        int    r;
        if constexpr ( std::is_same< float,T >::value ) {
            r = sposv_( uplo, &n, &nrhs, m_lapA, &lda, m_lapAinv, &ldb, &info );
        }
        else {
            r = dposv_( uplo, &n, &nrhs, m_lapA, &lda, m_lapAinv, &ldb, &info );
        }
        if ( r != 0 ) {
            std::cerr << "sposv returned non zeror:" << r << " info:"  << info << "\n";
        }

    }
};

#endif /*__TEST_CASE_CHOLESKY_LAPACK_INVERSE_H__*/
