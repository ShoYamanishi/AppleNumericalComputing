#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#include "test_case_cholesky_gsl.h"

template<>
TestCaseCholesky_gsl<double, true>::TestCaseCholesky_gsl( const int dim )
    :TestCaseCholesky<double, true>( dim )
    ,m_Ap(nullptr)
    ,m_xp(nullptr)
    ,m_bp(nullptr)
{
    this->setImplementationType( GSL );

    m_Ap = gsl_matrix_alloc( dim, dim );
    if ( m_Ap == nullptr ) {
        cerr << "failed to allocate with gsl_matrix_alloc()\n";
    }
    //cerr << "Alignment of Ap_:" << ((gsl_matrix*)Ap_)->tda << "\n";
    
    m_xp = gsl_vector_alloc( dim );
    if ( m_xp == nullptr ) {
        cerr << "failed to allocate with gsl_vector_alloc() ckp1\n";
    }
    //cerr << "Alignment of xp_:" << ((gsl_vector*)xp_)->stride << "\n";

    m_bp = gsl_vector_alloc( dim );
    if ( m_bp == nullptr ) {
        cerr << "failed to allocate with gsl_vector_alloc() ckp2\n";
    }
    //cerr << "Alignment of bp_:" << ((gsl_vector*)bp_)->stride << "\n";
}

template<>
TestCaseCholesky_gsl<double, true>::~TestCaseCholesky_gsl()
{
    if ( m_Ap != nullptr ) {
        gsl_matrix_free( (gsl_matrix*)m_Ap );
    }

    if ( m_xp != nullptr ) {
        gsl_vector_free( (gsl_vector*)m_xp );
    }

    if ( m_bp != nullptr ) {
        gsl_vector_free( (gsl_vector*)m_bp );
    }
}

template<>
void TestCaseCholesky_gsl<double, true>::setInitialStates( double* A, double* b )
{
    for ( int i = 0; i < this->m_dim; i++ ) {

        for ( int j = 0; j <= i; j++ ) {

            const double val = A[ lower_mat_index<true>( i, j, this->m_dim ) ];

            // Ap is column-major
            gsl_matrix_set( (gsl_matrix*)m_Ap, i, j, val );

            if ( i != j ) {
                gsl_matrix_set( (gsl_matrix*)m_Ap, j, i, val );
            }
        }

        gsl_vector_set( (gsl_vector*)m_bp, i, b[i] );
    }

    TestCaseCholesky< double, true >::setInitialStates( A, b );
}

template<>
void TestCaseCholesky_gsl<double, true>::compareTruth( const double* const L_baseline, const double* const x_baseline )
{
    for ( int i = 0; i < this->m_dim; i++ ) {

        this->m_x[i] = gsl_vector_get( (gsl_vector*)m_xp, i );

        for ( int j = 0; j <= i; j++ ) {

            this->m_L[ lower_mat_index<true>( i, j, this->m_dim ) ] = gsl_matrix_get( (gsl_matrix*)m_Ap, i, j );
        }
    }

    TestCaseCholesky<double,true>::compareTruth( L_baseline, x_baseline );
}

template<>
void TestCaseCholesky_gsl<double, true>::run()
{
    int r1 = gsl_linalg_cholesky_decomp1( (gsl_matrix*)m_Ap );
    if ( r1 != 0 ) {
        cerr << "gel_linalg_cholesky_decomp1(): " << r1 << "\n";
    }

    int r2 = gsl_linalg_cholesky_solve( (gsl_matrix*)m_Ap, (gsl_vector*)m_bp, (gsl_vector*)m_xp );
    if ( r2 != 0 ) {
        cerr << "gel_linalg_cholesky_solve(): " << r2 << "\n";
    }
}

