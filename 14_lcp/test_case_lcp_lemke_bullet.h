#ifndef __TEST_CASE_LCP_BULLET_H__
#define __TEST_CASE_LCP_BULLET_H__
#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include "btLemkeAlgorithm.h"
#include "btMatrixX.h"

#include "test_case_lcp.h"

template<class T, bool IS_COL_MAJOR>
class TestCaseLCP_lemke_bullet : public TestCaseLCP<T, IS_COL_MAJOR> {

    int        m_max_num_iterations;
    btMatrixXu m_bullet_M;
    btVectorXu m_bullet_q;
    bool       m_use_improved_lexico_minimum;
  public:

    TestCaseLCP_lemke_bullet( const int dim, const T condition_num, const int max_num_iterations, const float epsilon, const bool use_improved_lexico_minimum, const LCPTestPatternType p_type )
        :TestCaseLCP<T, IS_COL_MAJOR>( dim, condition_num, epsilon, p_type )
        ,m_max_num_iterations ( max_num_iterations )
        ,m_bullet_M( dim, dim )
        ,m_bullet_q( dim )
        ,m_use_improved_lexico_minimum( use_improved_lexico_minimum )
    {
        if constexpr (IS_COL_MAJOR) {
            assert(true); //column major not supported for Lemke.
        }
        if constexpr ( std::is_same< double,T >::value ) {
            assert(true); //double version cannot be implemented at the same time, as it uses preprocessor macros, not templates.
        }
        if ( use_improved_lexico_minimum ) {
            this->setImplementationType( LEMKE_BULLET_IMPROVED_LEXICO_MINIMUM );
        }
        else {
            this->setImplementationType( LEMKE_BULLET_ORIGINAL );
        }
    }

    virtual ~TestCaseLCP_lemke_bullet(){

    }

    virtual void prologue() {
        // Fill mM and mq here before run()
        for ( int row = 0; row < this->m_dim; row++ ) {

            for ( int col = 0; col < this->m_dim; col++ ) {

                m_bullet_M.setElem( row, col, this->m_M[ row * this->m_dim + col ] );

            }
            
            m_bullet_q[row] =  this->m_q[ row ];
        }
    }

    virtual void run() {

        btLemkeAlgorithm bt( m_bullet_M, m_bullet_q );

        btVectorXu zw = bt.solve( m_max_num_iterations, m_use_improved_lexico_minimum );

        for ( int i = 0; i < this->m_dim; i++ ) {
            this->m_w[i] = zw[i];
        }
        for ( int i = 0; i < this->m_dim; i++ ) {
            this->m_z[i] = zw[ i+this->m_dim ];
        }

        this->setIterations( bt.getSteps(), 0, 0 );
    }

    virtual void epilogue() {

    }
};

#endif /*__TEST_CASE_LCP_BULLET_H__*/
