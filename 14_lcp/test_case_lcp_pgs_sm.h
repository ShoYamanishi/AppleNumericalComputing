#ifndef __TEST_CASE_LCP_PGS_SM_H__
#define __TEST_CASE_LCP_PGS_SM_H__
#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <vector>
#include <Accelerate/Accelerate.h>

#include "test_case_lcp.h"


template<class T, bool IS_COL_MAJOR>
class TestCaseLCP_pgs_sm : public TestCaseLCP<T, IS_COL_MAJOR> {

    // Type of problem that can be solved:
    //
    //   M z + q = w
    //
    //   s.t. 0 <= z cmpl. w >= 0
    //
    //   M must be symmetric ( For the precise list of the types of matrices accepted.
    //
    //   The iteration formula
    //   z^{r+1} = - (q + L z^{r+1} + U z^r) / D

    const int           m_max_num_iterations;

    std::vector<double> m_error_history;

    T*                  m_z_lo;
    T*                  m_z_hi;
    T*                  m_z_abs;
    T*                  m_w_abs;
    T*                  m_Mz;
    T*                  m_active_M;
    T*                  m_active_q_z;
    int*                m_active_indices;
    int*                m_inactive_indices;
    int                 m_num_active_indices;
    int                 m_num_inactive_indices;
    int                 m_num_pgs;

  public:

    TestCaseLCP_pgs_sm( const int dim, const T condition_num, const int max_num_iterations, const T epsilon, const int num_pgs, const LCPTestPatternType p_type )
        :TestCaseLCP<T, IS_COL_MAJOR>( dim, condition_num, epsilon, p_type )
        ,m_max_num_iterations   ( max_num_iterations )
        ,m_num_active_indices   ( 0 )
        ,m_num_inactive_indices ( 0 )
        ,m_num_pgs              ( num_pgs )
    {
        static_assert( is_same< float,T >::value || is_same< double,T >::value );

        this->setImplementationType( PGS_SM_VDSP );

        m_z_lo = new T[ this->m_dim ];
        m_z_hi = new T[ this->m_dim ];

        memset( m_z_lo, 0, sizeof(T)*this->m_dim );

        for ( int i = 0; i < this->m_dim; i++ ){

            m_z_hi[i] = std::numeric_limits<T>::max();
        }

        m_z_abs            = new T[ this->m_dim ];
        m_w_abs            = new T[ this->m_dim ];
        m_active_M         = new T  [ this->m_dim * this->m_dim ];
        m_active_q_z       = new T  [ this->m_dim ];
        m_active_indices   = new int[ this->m_dim ];
        m_inactive_indices = new int[ this->m_dim ];
        m_Mz               = new T  [ this->m_dim ];
    }

    virtual ~TestCaseLCP_pgs_sm(){

        delete[] m_z_abs;
        delete[] m_w_abs;
        delete[] m_z_lo;
        delete[] m_z_hi;
        delete[] m_active_M;
        delete[] m_active_q_z;
        delete[] m_active_indices;
        delete[] m_inactive_indices;
        delete[] m_Mz;
    }

    virtual void setBoxConstraints( const T* lo, const T*hi ) {

        memcpy( m_z_lo, lo, sizeof(T)*this->m_dim );
        memcpy( m_z_hi, hi, sizeof(T)*this->m_dim );
    }

    virtual T* getActiveZ() {

        return this->m_z;
    }

    virtual void run() {

        this->m_error_history.clear();

        memset( this->m_z, 0, sizeof(T) * this->m_dim );

        int  num_iterations;
        int  num_pgs = 0;
        int  num_subspace_minimizations = 0;

        for ( num_iterations = 0; num_iterations < m_max_num_iterations; num_iterations++ ) {

            if constexpr ( is_same<float, T>::value ) {
                for( int i = 0; i < m_num_pgs/2; i++ ) {
                    num_pgs++;
                    calcZ_f();
                }
            }
            else {
                for( int i = 0; i < m_num_pgs/2; i++ ) {
                    num_pgs++;
                    calcZ_d();
                }
            }

            // sort the variables into active set and inactive set.
            int active_index   = 0;
            int inactive_index = 0;
            for( int i = 0; i < this->m_dim; i++ ) {
                if ( fabs(this->m_z[i] - m_z_lo[i]) < this->m_epsilon ) {
                    this->m_z[i] = m_z_lo[i];
                    m_inactive_indices[inactive_index] = i;
                    inactive_index++;
                }
                else if ( fabs(this->m_z[i] - m_z_hi[i]) < this->m_epsilon ) {
                    this->m_z[i] = m_z_hi[i];
                    m_inactive_indices[inactive_index] = i;
                    inactive_index++;
                }
                else {
                    // z[i] is active.
                    m_active_indices[active_index] = i;
                    active_index++;
                }
            }
            m_num_active_indices = active_index;
            m_num_inactive_indices = inactive_index;

            if (m_num_active_indices > 0) {

                num_subspace_minimizations++;

                // assemble active matrix;
                for ( int i = 0; i < m_num_active_indices; i++ ) {

                    const int active_i = m_active_indices[i];

                    for ( int j = 0; j < m_num_active_indices; j++ ) {

                        const int active_j = m_active_indices[j];

                        m_active_M[i*m_num_active_indices + j] = this->m_M[ active_i * this->m_dim + active_j ];

                    }
                }

                // assemble active q;
                for ( int i = 0; i < m_num_active_indices; i++ ) {

                    const int active_i = m_active_indices[i];

                    T inactive_Mz = 0.0;
                    for ( int j = 0; j < m_num_inactive_indices; j++ ) {
                       
                        const int inactive_j = m_inactive_indices[j];

                        inactive_Mz += (this->m_M[ active_i * this->m_dim + inactive_j ]*this->m_z[inactive_j]);
                    }
                    m_active_q_z[i] = -1.0 * ( this->m_q[active_i] + inactive_Mz );
                }

                // find active_z with Cholesky factorization.
                char   uplo[2] = "L";
                int    n       = m_num_active_indices;
                int    nrhs    = 1;
                int    lda     = m_num_active_indices;
                int    ldb     = m_num_active_indices;
                int    info;
                int    r;
                if constexpr ( is_same<float, T>::value ) {
                    r = sposv_( uplo, &n, &nrhs, m_active_M, &lda, m_active_q_z, &ldb, &info );
                }
                else {
                    r = dposv_( uplo, &n, &nrhs, m_active_M, &lda, m_active_q_z, &ldb, &info );
                }

                if ( r != 0 ) {
                    std::cerr << "sposv returned non zeror:" << r << " info:"  << info << "\n";
                }
                else {

                    int num_changes = 0;

                    // inspect the previsously active variables.
                    for ( int i = 0; i < m_num_active_indices; i++ ) {

                        const int active_i = m_active_indices[i];

                        if ( m_active_q_z[i] < m_z_lo[active_i] - this->m_epsilon ) {
                            this->m_z[active_i] = m_z_lo[active_i];
                            num_changes++;
                        }
                        else if ( m_active_q_z[i] > m_z_hi[active_i] + this->m_epsilon ) {
                            this->m_z[active_i] = m_z_hi[active_i];
                            num_changes++; 
                        }
                        else {
                            // z[i] is active.
                            this->m_z[active_i] = m_active_q_z[i];
                        }
                    }
                    if (num_changes == 0) {

                        // run PGS before return.
                        T error;
                        if constexpr ( is_same<float, T>::value ) {
                            for( int i = 0; i < m_num_pgs/2; i++ ) {
                                num_pgs++;
                                calcZ_f();
                            }
                            error = getErrorvDSP_f();
                        }
                        else {
                            for( int i = 0; i < m_num_pgs/2; i++ ) {
                                num_pgs++;
                                calcZ_d();
                            }
                            error = getErrorvDSP_d();
                        }
                        m_error_history.push_back( error );
                        break;
                    }
                }
                if constexpr ( is_same<float, T>::value ) {
                    for( int i = 0; i < m_num_pgs/2; i++ ) {
                        num_pgs++;
                        calcZ_f();
                    }
                }
                else {
                    for( int i = 0; i < m_num_pgs/2; i++ ) {
                        num_pgs++;
                        calcZ_d();
                    }
                }
            }

            T error;
            if constexpr ( is_same<float, T>::value ) {
                error = getErrorvDSP_f();
            }
            else {
                error = getErrorvDSP_d();
            }
            m_error_history.push_back( error );

            if (num_iterations >= 5 ) {

                double past_5 = 0.0;
                for ( int i = num_iterations - 5 ; i < num_iterations ; i++ ) {
                    past_5 += m_error_history[i];
                }
                past_5 /= 5.0;
                if ( error >= past_5 - this->m_epsilon ) {
                    // converged.
                    break;
                }
            }

        }
        this->setIterations( 0, num_pgs, num_subspace_minimizations );
    }

    double getErrorvDSP_f() {

        for ( int row = 0; row < this->m_dim; row++ ) {
            vDSP_dotpr ( &(this->m_M[ row * this->m_dim ]), 1, this->m_z, 1, &(this->m_Mz[row]), this->m_dim );
        }
        vDSP_vadd( this->m_Mz, 1, this->m_q, 1, this->m_w, 1, this->m_dim );

        T violation = 0.0;

        for ( int i = 0; i < this->m_dim; i++ ) {

            const T z = this->m_z[i];
            const T w = this->m_w[i];
            const T L = m_z_lo[i];
            const T H = m_z_hi[i];

            if ( fabs(L - 0.0) < this->m_epsilon ) {
                // normal constraint
                violation += ( fabs(z*w) - std::min(z, (T)0.0) - std::min(w, (T)0.0) );
            }
            else if (z > (L + H ) / 2.0 ) {
                violation += ( fabs( ( z - H ) * w ) + std::max( z - H, (T)0.0) );
            }
            else {
                violation += ( fabs( ( L - z ) * w ) + std::max( L - z, (T)0.0) );
            }
        }
        //cerr << "violation: " << violation << "\n";
        return violation;
    }

    double getErrorvDSP_d() {

        for ( int row = 0; row < this->m_dim; row++ ) {
            vDSP_dotprD ( &(this->m_M[ row * this->m_dim ]), 1, this->m_z, 1, &(this->m_Mz[row]), this->m_dim );
        }
        vDSP_vaddD( this->m_Mz, 1, this->m_q, 1, this->m_w, 1, this->m_dim );

        T violation = 0.0;

        for ( int i = 0; i < this->m_dim; i++ ) {

            const T z = this->m_z[i];
            const T w = this->m_w[i];
            const T L = m_z_lo[i];
            const T H = m_z_hi[i];

            if ( fabs(L - 0.0) < this->m_epsilon ) {
                // normal constraint
                violation += ( fabs(z*w) - std::min(z, (T)0.0) - std::min(w, (T)0.0) );
            }
            else if (z > (L + H ) / 2.0 ) {
                violation += ( fabs( ( z - H ) * w ) + std::max( z - H, (T)0.0) );
            }
            else {
                violation += ( fabs( ( L - z ) * w ) + std::max( L - z, (T)0.0) );
            }
        }
        //cerr << "violation: " << violation << "\n";
        return violation;
    }

    inline void calcZ_f() {

        for ( int row = 0; row < this->m_dim; row++ ) {

            // calc z^{r+1} = - (q + L z^{r+1} + U z^r) / D
            T diag = this->m_M[ row * this->m_dim + row ];
            T dot;
            vDSP_dotpr ( &(this->m_M[ row * this->m_dim ]), 1, this->m_z, 1, &dot, this->m_dim );

            this->m_z[row] = min ( max ( (diag * this->m_z[row] - dot - this->m_q[row]) / diag, m_z_lo[row] ), m_z_hi[row] );
        }
    }

    inline void calcZ_d() {

        for ( int row = 0; row < this->m_dim; row++ ) {

            // calc z^{r+1} = - (q + L z^{r+1} + U z^r) / D
            T diag = this->m_M[ row * this->m_dim + row ];
            T dot;
            vDSP_dotprD ( &(this->m_M[ row * this->m_dim ]), 1, this->m_z, 1, &dot, this->m_dim );

            this->m_z[row] = min ( max ( (diag * this->m_z[row] - dot - this->m_q[row]) / diag, m_z_lo[row] ), m_z_hi[row] );
        }
    }
};

#endif /*__TEST_CASE_LCP_PGS_SM_H__*/
