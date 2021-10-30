#ifndef __TEST_CASE_CHOLESKY_BASELINE_H__
#define __TEST_CASE_CHOLESKY_BASELINE_H__

#include "test_case_cholesky.h"


template<class T, bool IS_COL_MAJOR>
static void factorize_in_place_submatrix_cholesky( T* A, const int dim )
{
    for ( int k = 0; k < dim; k++ ) {

        const T A_kk = sqrt( A[ lower_mat_index<IS_COL_MAJOR>( k, k, dim ) ] );

        A[ lower_mat_index<IS_COL_MAJOR>( k, k, dim ) ] = A_kk;

        for ( int i = k + 1; i < dim ; i++ ) {

            A[ lower_mat_index<IS_COL_MAJOR>( i, k, dim ) ] /= A_kk;
        }

        for ( int j = k + 1; j < dim ; j++ ) {

            for ( int i = j; i < dim; i++ ) {

                A[ lower_mat_index<IS_COL_MAJOR>( i, j, dim ) ]
                    -= ( A[ lower_mat_index<IS_COL_MAJOR>( i, k, dim ) ] * A[ lower_mat_index<IS_COL_MAJOR>( j, k, dim )] );
            }
        }
    }
}


template<class T, bool IS_COL_MAJOR>
static void factorize_in_place_column_cholesky( T* A, const int dim )
{
    for ( int j = 0; j < dim; j++ ) {

        for ( int k = 0 ; k < j - 1; k++ ) {

            for ( int i = j; i < dim; i++ ) {

                A[ lower_mat_index<IS_COL_MAJOR>( i, j, dim ) ]

                    -= ( A[ lower_mat_index<IS_COL_MAJOR>( i, k, dim ) ] * A[ lower_mat_index<IS_COL_MAJOR>( j, k, dim ) ] );
            }
        }

        const T A_jj = sqrt( A[ lower_mat_index<IS_COL_MAJOR>( j, j, dim ) ] );
        A[ lower_mat_index<IS_COL_MAJOR>( j, j, dim ) ] = A_jj;

        for ( int i = j + 1; i < dim ; i++ ) {

            A[ lower_mat_index<IS_COL_MAJOR>( i, j, dim ) ] /= A_jj;
        }
    }
}


template<class T, bool IS_COL_MAJOR>
static T check_cholesky_factorization( const T* const A, const T* const L, const int dim )
{
    T accum_error = 0.0;

    for ( int i = 0; i < dim ; i++ ) {

        for ( int j = 0; j <= i ; j++ ) {

            T LLt_ij = 0.0;

            for ( int k = 0; k < dim; k++ ) {

                T L_ik  = (i >= k) ? L[ lower_mat_index<IS_COL_MAJOR>( i, k, dim ) ] : 0.0;
                T Lt_kj = (j >= k) ? L[ lower_mat_index<IS_COL_MAJOR>( j, k, dim ) ] : 0.0; // Lt_kj = L_jk

                LLt_ij += ( L_ik * Lt_kj );
            }

            accum_error += fabs( A[ lower_mat_index<IS_COL_MAJOR>( i, j, dim )] - LLt_ij );
        }
    }
    return accum_error / ((T)(dim * dim));
}


template<class T, bool IS_COL_MAJOR>
static void solve_Lxey( const T* const L, T* const x, const T* const y, const int dim )
{
    for ( int i = 0; i < dim; i++ ) {

        T sum = 0.0;

        for ( int k = 0; k < i; k++ ) {
            sum += ( L[ lower_mat_index<IS_COL_MAJOR>( i, k, dim ) ] * x[k] );
        }

        x[i] = ( y[i] - sum )/ L[ lower_mat_index<IS_COL_MAJOR>( i, i, dim ) ];
    }
}


template<class T, bool IS_COL_MAJOR>
static void solve_Ltxey( const T* const L, T* const x, const T* const y, const int dim )
{
    for ( int i = dim - 1; i >= 0; i-- ) {

        T sum = 0.0;

        for ( int k = dim - 1; k > i; k-- ) {
            sum += ( L[ lower_mat_index<IS_COL_MAJOR>( k, i, dim ) ] * x[k] ); // Lt_ik = L_ki
        }

        x[i] = ( y[i] - sum )/ L[ lower_mat_index<IS_COL_MAJOR>( i, i, dim ) ];
    }
}


template<class T, bool IS_COL_MAJOR>
class TestCaseCholesky_baseline : public TestCaseCholesky<T, IS_COL_MAJOR> {

  protected:
    const bool m_use_column_cholesky;

  public:

    TestCaseCholesky_baseline( const int dim, const bool use_column_cholesky )
        :TestCaseCholesky<T, IS_COL_MAJOR>( dim )
        ,m_use_column_cholesky( use_column_cholesky )
    {
        if ( m_use_column_cholesky ) {
            this->setColumnCholesky();
        }
        else {
            this->setSubmatrixCholesky();
        }
    }

    virtual ~TestCaseCholesky_baseline(){;}

    virtual void run() {

        if ( m_use_column_cholesky ) {
            factorize_in_place_column_cholesky<T, IS_COL_MAJOR>( this->m_L, this->m_dim );
        }
        else {
            factorize_in_place_submatrix_cholesky<T, IS_COL_MAJOR>( this->m_L, this->m_dim );
        }
        solve_Lxey <T, IS_COL_MAJOR>( this->m_L, this->m_y, this->m_b, this->m_dim );
        solve_Ltxey<T, IS_COL_MAJOR>( this->m_L, this->m_x, this->m_y, this->m_dim );
    }
};

#endif /*__TEST_CASE_CHOLESKY_BASELINE_H__*/
