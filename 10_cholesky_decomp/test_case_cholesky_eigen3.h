#ifndef __TEST_CASE_CHOLESKY_EIGEN3_H__
#define __TEST_CASE_CHOLESKY_EIGEN3_H__

#include "test_case_cholesky.h"

#include <Eigen/Cholesky>

template<class T, bool IS_COL_MAJOR>
class TestCaseCholesky_eigen3 : public TestCaseCholesky<T, IS_COL_MAJOR> {

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_eA;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_eL;
    Eigen::Matrix<T, Eigen::Dynamic, 1>              m_ex;
    Eigen::Matrix<T, Eigen::Dynamic, 1>              m_eb;

  public:

    TestCaseCholesky_eigen3( const int dim )
        :TestCaseCholesky<T, IS_COL_MAJOR>( dim )
    {
        m_eA.resize( dim, dim );
        m_eL.resize( dim, dim );
        m_ex.resize( dim, 1   );
        m_eb.resize( dim, 1   );

        this->setImplementationType(EIGEN3 );
    }

    virtual ~TestCaseCholesky_eigen3(){;}

    virtual void setInitialStates( T* A, T* b ) {

        for ( int i = 0; i < this->m_dim; i++ ) {

            for ( int j = 0; j <= i; j++ ) {

                const T val = A[ lower_mat_index<IS_COL_MAJOR>( i, j, this->m_dim ) ];

                m_eA( i, j ) = val;

                if ( i != j ) { 

                    m_eA( j, i ) = val;
                }
            }
            m_eb( i, 0 ) = b[i];
        }

        TestCaseCholesky<T,IS_COL_MAJOR>::setInitialStates( A, b );
    }

    virtual void compareTruth( const T* const L_baseline, const T* const x_baseline ) {

        for ( int i = 0; i < this->m_dim; i++ ) {

            this->m_x[i] = m_ex(i);

            for ( int j = 0; j <= i; j++ ) {
                this->m_L[ lower_mat_index<IS_COL_MAJOR>( i, j, this->m_dim ) ] = m_eL( i, j );
            }
        }
        return TestCaseCholesky<T,IS_COL_MAJOR>::compareTruth( L_baseline, x_baseline );
    }

    void run() {

        Eigen::LLT< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> , Eigen::Lower > llt( m_eA );

        m_eL = llt.matrixL();
        m_ex = llt.solve( m_eb );
    }
};


#endif /*__TEST_CASE_CHOLESKY_EIGEN3_H__*/
