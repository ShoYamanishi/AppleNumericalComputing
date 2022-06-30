#ifndef __TEST_CASE_LCP_PGS_H__
#define __TEST_CASE_LCP_PGS_H__
#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <Accelerate/Accelerate.h>

#include "test_case_lcp.h"


template<class T, bool IS_COL_MAJOR>
class TestCaseLCP_pgs : public TestCaseLCP<T, IS_COL_MAJOR> {

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

  public:

    TestCaseLCP_pgs( const int dim, const T condition_num, const int max_num_iterations, const T epsilon, const LCPTestPatternType p_type )
        :TestCaseLCP<T, IS_COL_MAJOR>( dim, condition_num, epsilon, p_type )
        ,m_max_num_iterations ( max_num_iterations )
        ,m_z_lo               ( nullptr )
        ,m_z_hi               ( nullptr )
    {
        static_assert( is_same< float,T >::value || is_same< double,T >::value );

        this->setImplementationType( PGS_VDSP );

        m_z_lo = new T[ this->m_dim ];
        m_z_hi = new T[ this->m_dim ];

        memset( m_z_lo, 0, sizeof(T)*this->m_dim );

        for ( int i = 0; i < this->m_dim; i++ ){

            m_z_hi[i] = std::numeric_limits<T>::max();
        }

        m_z_abs = new T[ this->m_dim ];
        m_w_abs = new T[ this->m_dim ];
        m_Mz    = new T[ this->m_dim ];
    }

    virtual ~TestCaseLCP_pgs(){

        delete[] m_z_abs;
        delete[] m_w_abs;
        delete[] m_z_lo;
        delete[] m_z_hi;
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
        for ( num_iterations = 0; num_iterations < m_max_num_iterations; num_iterations++ ) {

            if constexpr ( is_same<float, T>::value ) {
                calcZ_f();
            }
            else {
                calcZ_d();
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

        this->setIterations( 0, num_iterations, 0 );
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

#endif /*__TEST_CASE_LCP_PGS_H__*/
