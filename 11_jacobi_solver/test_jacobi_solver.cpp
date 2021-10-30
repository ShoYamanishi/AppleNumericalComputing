#include <Accelerate/Accelerate.h>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"

#include "jacobi_solver_metal_cpp.h"


template< class T, bool IS_COL_MAJOR  >
class TestCaseJacobiSolver : public TestCaseWithTimeMeasurements {

  protected:
    const int      m_dim;
    const int      m_iteration;
    T*             m_A;
    T*             m_D;
    T*             m_b;
    T*             m_x1;
    T*             m_x2;

    bool           m_updating_x1;
    vector<double> m_diff_x1_x2;

  public:

    TestCaseJacobiSolver( const int dim, const int iteration )
        :m_dim                   ( dim       )
        ,m_iteration             ( iteration )
        ,m_A                     ( nullptr   )
        ,m_D                     ( nullptr   )
        ,m_b                     ( nullptr   )
        ,m_x1                    ( nullptr   )
        ,m_x2                    ( nullptr   )
        ,m_updating_x1           ( false     )
    {
        static_assert( is_same< float,T >::value || is_same< double,T >::value );

        if constexpr ( is_same<float, T>::value ) {

            setDataElementType( FLOAT );
        }
        else if constexpr ( is_same<double, T>::value ) {

            setDataElementType( DOUBLE );
        }
        else {
            assert(true);
        }

        if constexpr (IS_COL_MAJOR) {
            setMatrixColMajor( dim, dim );
        }
        else {
            setMatrixRowMajor( dim, dim );
        }

        setVerificationType( RMS );
    }

    virtual ~TestCaseJacobiSolver(){;}

    virtual void compareTruth( const T* const baseline )
    {
        auto rms  = getRMSDiffTwoVectors( getActiveX(), baseline, m_dim );
        this->setRMS( rms );
    }

    virtual void setInitialStates( T* A, T* D, T* b , T* x1, T* x2 )
    {
        m_A           = A;
        m_D           = D;
        m_b           = b;
        m_x1          = x1;
        m_x2          = x2;
        m_updating_x1 = false;
    }

    virtual void run() = 0;

    virtual T* getActiveX() {

        return m_updating_x1 ? m_x1 : m_x2 ;
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseJacobiSolver_baseline : public TestCaseJacobiSolver< T, IS_COL_MAJOR > {

    T* m_Dinv;

  public:

    TestCaseJacobiSolver_baseline( const int dim, const int solver_iteration )
        :TestCaseJacobiSolver< T, IS_COL_MAJOR >( dim, solver_iteration )
        ,m_Dinv( new T[dim] )
    {
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseJacobiSolver_baseline() {
        delete[] m_Dinv;
    }

    virtual void run() {

        this->m_diff_x1_x2.clear();

        for ( int i = 0; i < this->m_dim; i++ ) {
            m_Dinv[i] = 1.0 / this->m_D[i];
        }

        for ( int i = 0; i < this->m_iteration; i++ ) {

            if ( this->m_updating_x1 ) {
                calcX1();
            }
            else {
                calcX2();
            }

            //T err = getRMSDiffTwoVectors( this->m_x1, this->m_x2, this->m_dim );
            T err = getDistTwoVectors( this->m_x1, this->m_x2, this->m_dim );

            this->m_diff_x1_x2.push_back( err );

            this->m_updating_x1 = ! this->m_updating_x1;
        }
    }

    void calcX1() {

        for ( int i = 0; i < this->m_dim; i++ ) {

            T sum = 0.0;

            for ( int j = 0; j < this->m_dim; j++ ) {
                sum += ( this->m_A[ linear_index_mat<IS_COL_MAJOR>(i, j, this->m_dim, this->m_dim) ] * this->m_x2[j] );

 
            }

            this->m_x1[i] = (this->m_b[i] - sum) * m_Dinv[ i ];
        }
    }

    void calcX2() {

        for ( int i = 0; i < this->m_dim; i++ ) {

            T sum = 0.0;

            for ( int j = 0; j < this->m_dim; j++ ) {
                sum += ( this->m_A[ linear_index_mat<IS_COL_MAJOR>(i, j, this->m_dim, this->m_dim) ] * this->m_x1[j] );
            }

            this->m_x2[i] = (this->m_b[i] - sum) * m_Dinv[ i ];
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseJacobiSolver_NEON : public TestCaseJacobiSolver< T, IS_COL_MAJOR > {

  protected:
    T*  m_Dinv;
    int m_factor_loop_unrolling;

  public:

    TestCaseJacobiSolver_NEON( const int dim, const int solver_iteration, const int factor_loop_unrolling )
        :TestCaseJacobiSolver< T, IS_COL_MAJOR >( dim, solver_iteration )
        ,m_Dinv( new T[ dim ] )
        ,m_factor_loop_unrolling( factor_loop_unrolling )
    { 
        static_assert( is_same< float,T >::value ||  is_same< double,T >::value );
        this->setNEON( 1, m_factor_loop_unrolling );
    }

    virtual ~TestCaseJacobiSolver_NEON() {
        delete[] m_Dinv;
    }

    virtual void run() {

        this->m_diff_x1_x2.clear();


        if constexpr ( is_same< float,T >::value ) {

            if ( m_factor_loop_unrolling == 1 ) {

                for (int i = 0; i < this->m_dim; i += 4 ) {

                    const float32x4_t qw_d1       = { this->m_D[i  ], this->m_D[i+1], this->m_D[i+2], this->m_D[i+3] };
                    const float32x4_t qw_d_inv1_1 = vrecpeq_f32( qw_d1 );
                    const float32x4_t qw_d_inv2_1 = vmulq_f32( vrecpsq_f32( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );

                    memcpy(&(m_Dinv[i  ]), &qw_d_inv2_1, sizeof(float)*4);
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for (int i = 0; i < this->m_dim; i+=8 ) {

                    const float32x4_t qw_d1       = { this->m_D[i  ], this->m_D[i+1], this->m_D[i+2], this->m_D[i+3] };
                    const float32x4_t qw_d2       = { this->m_D[i+4], this->m_D[i+5], this->m_D[i+6], this->m_D[i+7] };
                    const float32x4_t qw_d_inv1_1 = vrecpeq_f32( qw_d1 );
                    const float32x4_t qw_d_inv1_2 = vrecpeq_f32( qw_d2 );
                    const float32x4_t qw_d_inv2_1 = vmulq_f32( vrecpsq_f32( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float32x4_t qw_d_inv2_2 = vmulq_f32( vrecpsq_f32( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );

                    memcpy(&(m_Dinv[i  ]), &qw_d_inv2_1, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+4]), &qw_d_inv2_2, sizeof(float)*4);
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for (int i = 0; i < this->m_dim; i += 16 ) {

                    const float32x4_t qw_d1       = { this->m_D[i   ], this->m_D[i+ 1], this->m_D[i+ 2], this->m_D[i+ 3] };
                    const float32x4_t qw_d2       = { this->m_D[i+ 4], this->m_D[i+ 5], this->m_D[i+ 6], this->m_D[i+ 7] };
                    const float32x4_t qw_d3       = { this->m_D[i+ 8], this->m_D[i+ 9], this->m_D[i+10], this->m_D[i+11] };
                    const float32x4_t qw_d4       = { this->m_D[i+12], this->m_D[i+13], this->m_D[i+14], this->m_D[i+15] };
                    const float32x4_t qw_d_inv1_1 = vrecpeq_f32( qw_d1 );
                    const float32x4_t qw_d_inv1_2 = vrecpeq_f32( qw_d2 );
                    const float32x4_t qw_d_inv1_3 = vrecpeq_f32( qw_d3 );
                    const float32x4_t qw_d_inv1_4 = vrecpeq_f32( qw_d4 );
                    const float32x4_t qw_d_inv2_1 = vmulq_f32( vrecpsq_f32( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float32x4_t qw_d_inv2_2 = vmulq_f32( vrecpsq_f32( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float32x4_t qw_d_inv2_3 = vmulq_f32( vrecpsq_f32( qw_d3, qw_d_inv1_3 ), qw_d_inv1_3 );
                    const float32x4_t qw_d_inv2_4 = vmulq_f32( vrecpsq_f32( qw_d4, qw_d_inv1_4 ), qw_d_inv1_4 );

                    memcpy(&(m_Dinv[i   ]), &qw_d_inv2_1, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+ 4]), &qw_d_inv2_2, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+ 8]), &qw_d_inv2_3, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+12]), &qw_d_inv2_4, sizeof(float)*4);
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for (int i = 0; i < this->m_dim; i += 32 ) {

                    const float32x4_t qw_d1       = { this->m_D[i   ], this->m_D[i+ 1], this->m_D[i+ 2], this->m_D[i+ 3] };
                    const float32x4_t qw_d2       = { this->m_D[i+ 4], this->m_D[i+ 5], this->m_D[i+ 6], this->m_D[i+ 7] };
                    const float32x4_t qw_d3       = { this->m_D[i+ 8], this->m_D[i+ 9], this->m_D[i+10], this->m_D[i+11] };
                    const float32x4_t qw_d4       = { this->m_D[i+12], this->m_D[i+13], this->m_D[i+14], this->m_D[i+15] };
                    const float32x4_t qw_d5       = { this->m_D[i+16], this->m_D[i+17], this->m_D[i+18], this->m_D[i+19] };
                    const float32x4_t qw_d6       = { this->m_D[i+20], this->m_D[i+21], this->m_D[i+22], this->m_D[i+23] };
                    const float32x4_t qw_d7       = { this->m_D[i+24], this->m_D[i+25], this->m_D[i+26], this->m_D[i+27] };
                    const float32x4_t qw_d8       = { this->m_D[i+28], this->m_D[i+29], this->m_D[i+30], this->m_D[i+31] };
                    const float32x4_t qw_d_inv1_1 = vrecpeq_f32( qw_d1 );
                    const float32x4_t qw_d_inv1_2 = vrecpeq_f32( qw_d2 );
                    const float32x4_t qw_d_inv1_3 = vrecpeq_f32( qw_d3 );
                    const float32x4_t qw_d_inv1_4 = vrecpeq_f32( qw_d4 );
                    const float32x4_t qw_d_inv1_5 = vrecpeq_f32( qw_d5 );
                    const float32x4_t qw_d_inv1_6 = vrecpeq_f32( qw_d6 );
                    const float32x4_t qw_d_inv1_7 = vrecpeq_f32( qw_d7 );
                    const float32x4_t qw_d_inv1_8 = vrecpeq_f32( qw_d8 );
                    const float32x4_t qw_d_inv2_1 = vmulq_f32( vrecpsq_f32( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float32x4_t qw_d_inv2_2 = vmulq_f32( vrecpsq_f32( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float32x4_t qw_d_inv2_3 = vmulq_f32( vrecpsq_f32( qw_d3, qw_d_inv1_3 ), qw_d_inv1_3 );
                    const float32x4_t qw_d_inv2_4 = vmulq_f32( vrecpsq_f32( qw_d4, qw_d_inv1_4 ), qw_d_inv1_4 );
                    const float32x4_t qw_d_inv2_5 = vmulq_f32( vrecpsq_f32( qw_d5, qw_d_inv1_5 ), qw_d_inv1_5 );
                    const float32x4_t qw_d_inv2_6 = vmulq_f32( vrecpsq_f32( qw_d6, qw_d_inv1_6 ), qw_d_inv1_6 );
                    const float32x4_t qw_d_inv2_7 = vmulq_f32( vrecpsq_f32( qw_d7, qw_d_inv1_7 ), qw_d_inv1_7 );
                    const float32x4_t qw_d_inv2_8 = vmulq_f32( vrecpsq_f32( qw_d8, qw_d_inv1_8 ), qw_d_inv1_8 );

                    memcpy(&(m_Dinv[i   ]), &qw_d_inv2_1, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+ 4]), &qw_d_inv2_2, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+ 8]), &qw_d_inv2_3, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+12]), &qw_d_inv2_4, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+16]), &qw_d_inv2_5, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+20]), &qw_d_inv2_6, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+24]), &qw_d_inv2_7, sizeof(float)*4);
                    memcpy(&(m_Dinv[i+28]), &qw_d_inv2_8, sizeof(float)*4);
                }
            }
        }
        else {
            if ( m_factor_loop_unrolling == 1 ) {

                for (int i = 0; i < this->m_dim; i+=2 ) {

                    const float64x2_t qw_d1       = { this->m_D[i  ], this->m_D[i+1] };
                    const float64x2_t qw_d_inv1_1 = vrecpeq_f64( qw_d1 );
                    const float64x2_t qw_d_inv2_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float64x2_t qw_d_inv3_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv2_1 ), qw_d_inv2_1 );

                    memcpy(&(m_Dinv[i  ]), &qw_d_inv3_1, sizeof(double)*2);
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for (int i = 0; i < this->m_dim; i+=4 ) {

                    const float64x2_t qw_d1       = { this->m_D[i  ], this->m_D[i+1] };
                    const float64x2_t qw_d2       = { this->m_D[i+2], this->m_D[i+3] };
                    const float64x2_t qw_d_inv1_1 = vrecpeq_f64( qw_d1 );
                    const float64x2_t qw_d_inv1_2 = vrecpeq_f64( qw_d2 );
                    const float64x2_t qw_d_inv2_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float64x2_t qw_d_inv2_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float64x2_t qw_d_inv3_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv2_1 ), qw_d_inv2_1 );
                    const float64x2_t qw_d_inv3_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv2_2 ), qw_d_inv2_2 );

                    memcpy(&(m_Dinv[i  ]), &qw_d_inv3_1, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+2]), &qw_d_inv3_2, sizeof(double)*2);
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for (int i = 0; i < this->m_dim; i+=8 ) {

                    const float64x2_t qw_d1       = { this->m_D[i  ], this->m_D[i+1] };
                    const float64x2_t qw_d2       = { this->m_D[i+2], this->m_D[i+3] };
                    const float64x2_t qw_d3       = { this->m_D[i+4], this->m_D[i+5] };
                    const float64x2_t qw_d4       = { this->m_D[i+6], this->m_D[i+7] };
                    const float64x2_t qw_d_inv1_1 = vrecpeq_f64( qw_d1 );
                    const float64x2_t qw_d_inv1_2 = vrecpeq_f64( qw_d2 );
                    const float64x2_t qw_d_inv1_3 = vrecpeq_f64( qw_d3 );
                    const float64x2_t qw_d_inv1_4 = vrecpeq_f64( qw_d4 );
                    const float64x2_t qw_d_inv2_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float64x2_t qw_d_inv2_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float64x2_t qw_d_inv2_3 = vmulq_f64( vrecpsq_f64( qw_d3, qw_d_inv1_3 ), qw_d_inv1_3 );
                    const float64x2_t qw_d_inv2_4 = vmulq_f64( vrecpsq_f64( qw_d4, qw_d_inv1_4 ), qw_d_inv1_4 );
                    const float64x2_t qw_d_inv3_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv2_1 ), qw_d_inv2_1 );
                    const float64x2_t qw_d_inv3_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv2_2 ), qw_d_inv2_2 );
                    const float64x2_t qw_d_inv3_3 = vmulq_f64( vrecpsq_f64( qw_d3, qw_d_inv2_3 ), qw_d_inv2_3 );
                    const float64x2_t qw_d_inv3_4 = vmulq_f64( vrecpsq_f64( qw_d4, qw_d_inv2_4 ), qw_d_inv2_4 );

                    memcpy(&(m_Dinv[i  ]), &qw_d_inv3_1, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+2]), &qw_d_inv3_2, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+4]), &qw_d_inv3_3, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+6]), &qw_d_inv3_4, sizeof(double)*2);
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for (int i = 0; i < this->m_dim; i+=16 ) {

                    const float64x2_t qw_d1       = { this->m_D[i   ], this->m_D[i+ 1] };
                    const float64x2_t qw_d2       = { this->m_D[i+ 2], this->m_D[i+ 3] };
                    const float64x2_t qw_d3       = { this->m_D[i+ 4], this->m_D[i+ 5] };
                    const float64x2_t qw_d4       = { this->m_D[i+ 6], this->m_D[i+ 7] };
                    const float64x2_t qw_d5       = { this->m_D[i+ 8], this->m_D[i+ 9] };
                    const float64x2_t qw_d6       = { this->m_D[i+10], this->m_D[i+11] };
                    const float64x2_t qw_d7       = { this->m_D[i+12], this->m_D[i+13] };
                    const float64x2_t qw_d8       = { this->m_D[i+14], this->m_D[i+15] };
                    const float64x2_t qw_d_inv1_1 = vrecpeq_f64( qw_d1 );
                    const float64x2_t qw_d_inv1_2 = vrecpeq_f64( qw_d2 );
                    const float64x2_t qw_d_inv1_3 = vrecpeq_f64( qw_d3 );
                    const float64x2_t qw_d_inv1_4 = vrecpeq_f64( qw_d4 );
                    const float64x2_t qw_d_inv1_5 = vrecpeq_f64( qw_d5 );
                    const float64x2_t qw_d_inv1_6 = vrecpeq_f64( qw_d6 );
                    const float64x2_t qw_d_inv1_7 = vrecpeq_f64( qw_d7 );
                    const float64x2_t qw_d_inv1_8 = vrecpeq_f64( qw_d8 );
                    const float64x2_t qw_d_inv2_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float64x2_t qw_d_inv2_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float64x2_t qw_d_inv2_3 = vmulq_f64( vrecpsq_f64( qw_d3, qw_d_inv1_3 ), qw_d_inv1_3 );
                    const float64x2_t qw_d_inv2_4 = vmulq_f64( vrecpsq_f64( qw_d4, qw_d_inv1_4 ), qw_d_inv1_4 );
                    const float64x2_t qw_d_inv2_5 = vmulq_f64( vrecpsq_f64( qw_d5, qw_d_inv1_5 ), qw_d_inv1_5 );
                    const float64x2_t qw_d_inv2_6 = vmulq_f64( vrecpsq_f64( qw_d6, qw_d_inv1_6 ), qw_d_inv1_6 );
                    const float64x2_t qw_d_inv2_7 = vmulq_f64( vrecpsq_f64( qw_d7, qw_d_inv1_7 ), qw_d_inv1_7 );
                    const float64x2_t qw_d_inv2_8 = vmulq_f64( vrecpsq_f64( qw_d8, qw_d_inv1_8 ), qw_d_inv1_8 );
                    const float64x2_t qw_d_inv3_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv2_1 ), qw_d_inv2_1 );
                    const float64x2_t qw_d_inv3_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv2_2 ), qw_d_inv2_2 );
                    const float64x2_t qw_d_inv3_3 = vmulq_f64( vrecpsq_f64( qw_d3, qw_d_inv2_3 ), qw_d_inv2_3 );
                    const float64x2_t qw_d_inv3_4 = vmulq_f64( vrecpsq_f64( qw_d4, qw_d_inv2_4 ), qw_d_inv2_4 );
                    const float64x2_t qw_d_inv3_5 = vmulq_f64( vrecpsq_f64( qw_d5, qw_d_inv2_5 ), qw_d_inv2_5 );
                    const float64x2_t qw_d_inv3_6 = vmulq_f64( vrecpsq_f64( qw_d6, qw_d_inv2_6 ), qw_d_inv2_6 );
                    const float64x2_t qw_d_inv3_7 = vmulq_f64( vrecpsq_f64( qw_d7, qw_d_inv2_7 ), qw_d_inv2_7 );
                    const float64x2_t qw_d_inv3_8 = vmulq_f64( vrecpsq_f64( qw_d8, qw_d_inv2_8 ), qw_d_inv2_8 );

                    memcpy(&(m_Dinv[i   ]), &qw_d_inv3_1, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+ 2]), &qw_d_inv3_2, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+ 4]), &qw_d_inv3_3, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+ 6]), &qw_d_inv3_4, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+ 8]), &qw_d_inv3_5, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+10]), &qw_d_inv3_6, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+12]), &qw_d_inv3_7, sizeof(double)*2);
                    memcpy(&(m_Dinv[i+14]), &qw_d_inv3_8, sizeof(double)*2);
                }
            }
        }

        for ( int i = 0; i < this->m_iteration; i++ ) {

            if ( this->m_updating_x1 ) {
                calcX1();
            }
            else {
                calcX2();
            }

            //T err = this->getRmsX1X2Neon();
            T err = this->getDistX1X2Neon();

            this->m_diff_x1_x2.push_back(err);

            this->m_updating_x1 = ! this->m_updating_x1;
        }
    }

    void calcX1() {
        if constexpr ( IS_COL_MAJOR ) {
            calcX1ColMajor( 0, this->m_dim );
        }
        else {
            calcX1RowMajor( 0, this->m_dim );
        }
    }

    void calcX2() {
        if constexpr ( IS_COL_MAJOR ) {
            calcX2ColMajor( 0, this->m_dim );
        }
        else {
            calcX2RowMajor( 0, this->m_dim );
        }
    }

    virtual void calcX1ColMajor( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            if ( m_factor_loop_unrolling == 1 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 4 ) {

                    float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const float       col_v   = this->m_x2[j];
                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float32x4_t qw_col  = { col_v, col_v, col_v, col_v };
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );

                        qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                    }

                    const float32x4_t qw_b1           = vld1q_f32( &(this->m_b[i  ]) );
                    const float32x4_t qw_Dinv1        = vld1q_f32( &(this->m_Dinv[i  ]) );
                    const float32x4_t qw_b_minus_sum1 = vsubq_f32( qw_b1, qw_row_sum1 );
                    const float32x4_t qw_x1_1         = vmulq_f32( qw_b_minus_sum1 , qw_Dinv1 );
                    memcpy( &(this->m_x1[i  ]), &qw_x1_1, sizeof(float)*4 );
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 8 ) {

                    float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const float       col_v   = this->m_x2[j];
                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ i + 4 + this->m_dim * j ]) );
                        const float32x4_t qw_col  = { col_v, col_v, col_v, col_v };
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );

                        qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                    }

                    const float32x4_t qw_b1           = vld1q_f32( &(this->m_b[i  ]) );
                    const float32x4_t qw_b2           = vld1q_f32( &(this->m_b[i+4]) );
                    const float32x4_t qw_Dinv1        = vld1q_f32( &(this->m_Dinv[i  ]) );
                    const float32x4_t qw_Dinv2        = vld1q_f32( &(this->m_Dinv[i+4]) );
                    const float32x4_t qw_b_minus_sum1 = vsubq_f32( qw_b1, qw_row_sum1 );
                    const float32x4_t qw_b_minus_sum2 = vsubq_f32( qw_b2, qw_row_sum2 );
                    const float32x4_t qw_x1_1         = vmulq_f32( qw_b_minus_sum1 , qw_Dinv1 );
                    const float32x4_t qw_x1_2         = vmulq_f32( qw_b_minus_sum2 , qw_Dinv2 );
                    memcpy( &(this->m_x1[i  ]), &qw_x1_1, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+4]), &qw_x1_2, sizeof(float)*4 );
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 16 ) {

                    float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum4 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const float       col_v   = this->m_x2[j];
                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ i      + this->m_dim * j ]) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ i +  4 + this->m_dim * j ]) );
                        const float32x4_t qw_mat3 = vld1q_f32( &(this->m_A[ i +  8 + this->m_dim * j ]) );
                        const float32x4_t qw_mat4 = vld1q_f32( &(this->m_A[ i + 12 + this->m_dim * j ]) );
                        const float32x4_t qw_col  = { col_v, col_v, col_v, col_v };
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );
                        const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col );
                        const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col );

                        qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                        qw_row_sum3 = vaddq_f32( qw_mc3, qw_row_sum3 );
                        qw_row_sum4 = vaddq_f32( qw_mc4, qw_row_sum4 );
                    }

                    const float32x4_t qw_b1           = vld1q_f32( &(this->m_b[i   ]) );
                    const float32x4_t qw_b2           = vld1q_f32( &(this->m_b[i+ 4]) );
                    const float32x4_t qw_b3           = vld1q_f32( &(this->m_b[i+ 8]) );
                    const float32x4_t qw_b4           = vld1q_f32( &(this->m_b[i+12]) );
                    const float32x4_t qw_Dinv1        = vld1q_f32( &(this->m_Dinv[i   ]) );
                    const float32x4_t qw_Dinv2        = vld1q_f32( &(this->m_Dinv[i+ 4]) );
                    const float32x4_t qw_Dinv3        = vld1q_f32( &(this->m_Dinv[i+ 8]) );
                    const float32x4_t qw_Dinv4        = vld1q_f32( &(this->m_Dinv[i+12]) );
                    const float32x4_t qw_b_minus_sum1 = vsubq_f32( qw_b1, qw_row_sum1 );
                    const float32x4_t qw_b_minus_sum2 = vsubq_f32( qw_b2, qw_row_sum2 );
                    const float32x4_t qw_b_minus_sum3 = vsubq_f32( qw_b3, qw_row_sum3 );
                    const float32x4_t qw_b_minus_sum4 = vsubq_f32( qw_b4, qw_row_sum4 );
                    const float32x4_t qw_x1_1         = vmulq_f32( qw_b_minus_sum1 , qw_Dinv1 );
                    const float32x4_t qw_x1_2         = vmulq_f32( qw_b_minus_sum2 , qw_Dinv2 );
                    const float32x4_t qw_x1_3         = vmulq_f32( qw_b_minus_sum3 , qw_Dinv3 );
                    const float32x4_t qw_x1_4         = vmulq_f32( qw_b_minus_sum4 , qw_Dinv4 );
                    memcpy( &(this->m_x1[i   ]), &qw_x1_1, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+ 4]), &qw_x1_2, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+ 8]), &qw_x1_3, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+12]), &qw_x1_4, sizeof(float)*4 );
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 32 ) {

                    float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum4 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum5 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum6 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum7 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum8 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const float       col_v   = this->m_x2[j];
                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ i      + this->m_dim * j ]) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ i +  4 + this->m_dim * j ]) );
                        const float32x4_t qw_mat3 = vld1q_f32( &(this->m_A[ i +  8 + this->m_dim * j ]) );
                        const float32x4_t qw_mat4 = vld1q_f32( &(this->m_A[ i + 12 + this->m_dim * j ]) );
                        const float32x4_t qw_mat5 = vld1q_f32( &(this->m_A[ i + 16 + this->m_dim * j ]) );
                        const float32x4_t qw_mat6 = vld1q_f32( &(this->m_A[ i + 20 + this->m_dim * j ]) );
                        const float32x4_t qw_mat7 = vld1q_f32( &(this->m_A[ i + 24 + this->m_dim * j ]) );
                        const float32x4_t qw_mat8 = vld1q_f32( &(this->m_A[ i + 28 + this->m_dim * j ]) );
                        const float32x4_t qw_col  = { col_v, col_v, col_v, col_v };
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );
                        const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col );
                        const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col );
                        const float32x4_t qw_mc5  = vmulq_f32( qw_mat5, qw_col );
                        const float32x4_t qw_mc6  = vmulq_f32( qw_mat6, qw_col );
                        const float32x4_t qw_mc7  = vmulq_f32( qw_mat7, qw_col );
                        const float32x4_t qw_mc8  = vmulq_f32( qw_mat8, qw_col );

                        qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                        qw_row_sum3 = vaddq_f32( qw_mc3, qw_row_sum3 );
                        qw_row_sum4 = vaddq_f32( qw_mc4, qw_row_sum4 );
                        qw_row_sum5 = vaddq_f32( qw_mc5, qw_row_sum5 );
                        qw_row_sum6 = vaddq_f32( qw_mc6, qw_row_sum6 );
                        qw_row_sum7 = vaddq_f32( qw_mc7, qw_row_sum7 );
                        qw_row_sum8 = vaddq_f32( qw_mc8, qw_row_sum8 );
                    }

                    const float32x4_t qw_b1           = vld1q_f32( &(this->m_b[i   ]) );
                    const float32x4_t qw_b2           = vld1q_f32( &(this->m_b[i+ 4]) );
                    const float32x4_t qw_b3           = vld1q_f32( &(this->m_b[i+ 8]) );
                    const float32x4_t qw_b4           = vld1q_f32( &(this->m_b[i+12]) );
                    const float32x4_t qw_b5           = vld1q_f32( &(this->m_b[i+16]) );
                    const float32x4_t qw_b6           = vld1q_f32( &(this->m_b[i+20]) );
                    const float32x4_t qw_b7           = vld1q_f32( &(this->m_b[i+24]) );
                    const float32x4_t qw_b8           = vld1q_f32( &(this->m_b[i+28]) );
                    const float32x4_t qw_Dinv1        = vld1q_f32( &(this->m_Dinv[i   ]) );
                    const float32x4_t qw_Dinv2        = vld1q_f32( &(this->m_Dinv[i+ 4]) );
                    const float32x4_t qw_Dinv3        = vld1q_f32( &(this->m_Dinv[i+ 8]) );
                    const float32x4_t qw_Dinv4        = vld1q_f32( &(this->m_Dinv[i+12]) );
                    const float32x4_t qw_Dinv5        = vld1q_f32( &(this->m_Dinv[i+16]) );
                    const float32x4_t qw_Dinv6        = vld1q_f32( &(this->m_Dinv[i+20]) );
                    const float32x4_t qw_Dinv7        = vld1q_f32( &(this->m_Dinv[i+24]) );
                    const float32x4_t qw_Dinv8        = vld1q_f32( &(this->m_Dinv[i+28]) );
                    const float32x4_t qw_b_minus_sum1 = vsubq_f32( qw_b1, qw_row_sum1 );
                    const float32x4_t qw_b_minus_sum2 = vsubq_f32( qw_b2, qw_row_sum2 );
                    const float32x4_t qw_b_minus_sum3 = vsubq_f32( qw_b3, qw_row_sum3 );
                    const float32x4_t qw_b_minus_sum4 = vsubq_f32( qw_b4, qw_row_sum4 );
                    const float32x4_t qw_b_minus_sum5 = vsubq_f32( qw_b5, qw_row_sum5 );
                    const float32x4_t qw_b_minus_sum6 = vsubq_f32( qw_b6, qw_row_sum6 );
                    const float32x4_t qw_b_minus_sum7 = vsubq_f32( qw_b7, qw_row_sum7 );
                    const float32x4_t qw_b_minus_sum8 = vsubq_f32( qw_b8, qw_row_sum8 );
                    const float32x4_t qw_x1_1         = vmulq_f32( qw_b_minus_sum1 , qw_Dinv1 );
                    const float32x4_t qw_x1_2         = vmulq_f32( qw_b_minus_sum2 , qw_Dinv2 );
                    const float32x4_t qw_x1_3         = vmulq_f32( qw_b_minus_sum3 , qw_Dinv3 );
                    const float32x4_t qw_x1_4         = vmulq_f32( qw_b_minus_sum4 , qw_Dinv4 );
                    const float32x4_t qw_x1_5         = vmulq_f32( qw_b_minus_sum5 , qw_Dinv5 );
                    const float32x4_t qw_x1_6         = vmulq_f32( qw_b_minus_sum6 , qw_Dinv6 );
                    const float32x4_t qw_x1_7         = vmulq_f32( qw_b_minus_sum7 , qw_Dinv7 );
                    const float32x4_t qw_x1_8         = vmulq_f32( qw_b_minus_sum8 , qw_Dinv8 );

                    memcpy( &(this->m_x1[i   ]), &qw_x1_1, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+ 4]), &qw_x1_2, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+ 8]), &qw_x1_3, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+12]), &qw_x1_4, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+16]), &qw_x1_5, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+20]), &qw_x1_6, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+24]), &qw_x1_7, sizeof(float)*4 );
                    memcpy( &(this->m_x1[i+28]), &qw_x1_8, sizeof(float)*4 );
                }
            }
        }
        else {
            if ( m_factor_loop_unrolling == 1 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 2 ) {

                    float64x2_t qw_row_sum1 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const double      col_v   = this->m_x2[j];
                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float64x2_t qw_col  = { col_v, col_v };
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );

                        qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                    }

                    const float64x2_t qw_b1           = vld1q_f64( &(this->m_b[i  ]) );
                    const float64x2_t qw_Dinv1        = vld1q_f64( &(this->m_Dinv[i  ]) );
                    const float64x2_t qw_b_minus_sum1 = vsubq_f64( qw_b1, qw_row_sum1 );
                    const float64x2_t qw_x1_1         = vmulq_f64( qw_b_minus_sum1 , qw_Dinv1 );
                    memcpy( &(this->m_x1[i  ]), &qw_x1_1, sizeof(double)*2 );
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 4 ) {

                    float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum2 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const double      col_v   = this->m_x2[j];
                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ i + 2 + this->m_dim * j ]) );
                        const float64x2_t qw_col  = { col_v, col_v };
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );

                        qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                    }

                    const float64x2_t qw_b1           = vld1q_f64( &(this->m_b[i  ]) );
                    const float64x2_t qw_b2           = vld1q_f64( &(this->m_b[i+2]) );
                    const float64x2_t qw_Dinv1        = vld1q_f64( &(this->m_Dinv[i  ]) );
                    const float64x2_t qw_Dinv2        = vld1q_f64( &(this->m_Dinv[i+2]) );
                    const float64x2_t qw_b_minus_sum1 = vsubq_f64( qw_b1, qw_row_sum1 );
                    const float64x2_t qw_b_minus_sum2 = vsubq_f64( qw_b2, qw_row_sum2 );
                    const float64x2_t qw_x1_1         = vmulq_f64( qw_b_minus_sum1 , qw_Dinv1 );
                    const float64x2_t qw_x1_2         = vmulq_f64( qw_b_minus_sum2 , qw_Dinv2 );
                    memcpy( &(this->m_x1[i  ]), &qw_x1_1, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+2]), &qw_x1_2, sizeof(double)*2 );
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 8 ) {

                    float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum2 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum3 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum4 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const double      col_v   = this->m_x2[j];
                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ i + 2 + this->m_dim * j ]) );
                        const float64x2_t qw_mat3 = vld1q_f64( &(this->m_A[ i + 4 + this->m_dim * j ]) );
                        const float64x2_t qw_mat4 = vld1q_f64( &(this->m_A[ i + 6 + this->m_dim * j ]) );
                        const float64x2_t qw_col  = { col_v, col_v };
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );
                        const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col );
                        const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col );

                        qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                        qw_row_sum3 = vaddq_f64( qw_mc3, qw_row_sum3 );
                        qw_row_sum4 = vaddq_f64( qw_mc4, qw_row_sum4 );
                    }

                    const float64x2_t qw_b1           = vld1q_f64( &(this->m_b[i  ]) );
                    const float64x2_t qw_b2           = vld1q_f64( &(this->m_b[i+2]) );
                    const float64x2_t qw_b3           = vld1q_f64( &(this->m_b[i+4]) );
                    const float64x2_t qw_b4           = vld1q_f64( &(this->m_b[i+6]) );
                    const float64x2_t qw_Dinv1        = vld1q_f64( &(this->m_Dinv[i  ]) );
                    const float64x2_t qw_Dinv2        = vld1q_f64( &(this->m_Dinv[i+2]) );
                    const float64x2_t qw_Dinv3        = vld1q_f64( &(this->m_Dinv[i+4]) );
                    const float64x2_t qw_Dinv4        = vld1q_f64( &(this->m_Dinv[i+6]) );
                    const float64x2_t qw_b_minus_sum1 = vsubq_f64( qw_b1, qw_row_sum1 );
                    const float64x2_t qw_b_minus_sum2 = vsubq_f64( qw_b2, qw_row_sum2 );
                    const float64x2_t qw_b_minus_sum3 = vsubq_f64( qw_b3, qw_row_sum3 );
                    const float64x2_t qw_b_minus_sum4 = vsubq_f64( qw_b4, qw_row_sum4 );
                    const float64x2_t qw_x1_1         = vmulq_f64( qw_b_minus_sum1 , qw_Dinv1 );
                    const float64x2_t qw_x1_2         = vmulq_f64( qw_b_minus_sum2 , qw_Dinv2 );
                    const float64x2_t qw_x1_3         = vmulq_f64( qw_b_minus_sum3 , qw_Dinv3 );
                    const float64x2_t qw_x1_4         = vmulq_f64( qw_b_minus_sum4 , qw_Dinv4 );
                    memcpy( &(this->m_x1[i  ]), &qw_x1_1, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+2]), &qw_x1_2, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+4]), &qw_x1_3, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+6]), &qw_x1_4, sizeof(double)*2 );
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 16 ) {

                    float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum2 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum3 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum4 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum5 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum6 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum7 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum8 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const double      col_v   = this->m_x2[j];
                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ i      + this->m_dim * j ]) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ i +  2 + this->m_dim * j ]) );
                        const float64x2_t qw_mat3 = vld1q_f64( &(this->m_A[ i +  4 + this->m_dim * j ]) );
                        const float64x2_t qw_mat4 = vld1q_f64( &(this->m_A[ i +  6 + this->m_dim * j ]) );
                        const float64x2_t qw_mat5 = vld1q_f64( &(this->m_A[ i +  8 + this->m_dim * j ]) );
                        const float64x2_t qw_mat6 = vld1q_f64( &(this->m_A[ i + 10 + this->m_dim * j ]) );
                        const float64x2_t qw_mat7 = vld1q_f64( &(this->m_A[ i + 12 + this->m_dim * j ]) );
                        const float64x2_t qw_mat8 = vld1q_f64( &(this->m_A[ i + 14 + this->m_dim * j ]) );
                        const float64x2_t qw_col  = { col_v, col_v };
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );
                        const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col );
                        const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col );
                        const float64x2_t qw_mc5  = vmulq_f64( qw_mat5, qw_col );
                        const float64x2_t qw_mc6  = vmulq_f64( qw_mat6, qw_col );
                        const float64x2_t qw_mc7  = vmulq_f64( qw_mat7, qw_col );
                        const float64x2_t qw_mc8  = vmulq_f64( qw_mat8, qw_col );

                        qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                        qw_row_sum3 = vaddq_f64( qw_mc3, qw_row_sum3 );
                        qw_row_sum4 = vaddq_f64( qw_mc4, qw_row_sum4 );
                        qw_row_sum5 = vaddq_f64( qw_mc5, qw_row_sum5 );
                        qw_row_sum6 = vaddq_f64( qw_mc6, qw_row_sum6 );
                        qw_row_sum7 = vaddq_f64( qw_mc7, qw_row_sum7 );
                        qw_row_sum8 = vaddq_f64( qw_mc8, qw_row_sum8 );
                    }

                    const float64x2_t qw_b1           = vld1q_f64( &(this->m_b[i   ]) );
                    const float64x2_t qw_b2           = vld1q_f64( &(this->m_b[i+ 2]) );
                    const float64x2_t qw_b3           = vld1q_f64( &(this->m_b[i+ 4]) );
                    const float64x2_t qw_b4           = vld1q_f64( &(this->m_b[i+ 6]) );
                    const float64x2_t qw_b5           = vld1q_f64( &(this->m_b[i+ 8]) );
                    const float64x2_t qw_b6           = vld1q_f64( &(this->m_b[i+10]) );
                    const float64x2_t qw_b7           = vld1q_f64( &(this->m_b[i+12]) );
                    const float64x2_t qw_b8           = vld1q_f64( &(this->m_b[i+14]) );
                    const float64x2_t qw_Dinv1        = vld1q_f64( &(this->m_Dinv[i   ]) );
                    const float64x2_t qw_Dinv2        = vld1q_f64( &(this->m_Dinv[i+ 2]) );
                    const float64x2_t qw_Dinv3        = vld1q_f64( &(this->m_Dinv[i+ 4]) );
                    const float64x2_t qw_Dinv4        = vld1q_f64( &(this->m_Dinv[i+ 6]) );
                    const float64x2_t qw_Dinv5        = vld1q_f64( &(this->m_Dinv[i+ 8]) );
                    const float64x2_t qw_Dinv6        = vld1q_f64( &(this->m_Dinv[i+10]) );
                    const float64x2_t qw_Dinv7        = vld1q_f64( &(this->m_Dinv[i+12]) );
                    const float64x2_t qw_Dinv8        = vld1q_f64( &(this->m_Dinv[i+14]) );
                    const float64x2_t qw_b_minus_sum1 = vsubq_f64( qw_b1, qw_row_sum1 );
                    const float64x2_t qw_b_minus_sum2 = vsubq_f64( qw_b2, qw_row_sum2 );
                    const float64x2_t qw_b_minus_sum3 = vsubq_f64( qw_b3, qw_row_sum3 );
                    const float64x2_t qw_b_minus_sum4 = vsubq_f64( qw_b4, qw_row_sum4 );
                    const float64x2_t qw_b_minus_sum5 = vsubq_f64( qw_b5, qw_row_sum5 );
                    const float64x2_t qw_b_minus_sum6 = vsubq_f64( qw_b6, qw_row_sum6 );
                    const float64x2_t qw_b_minus_sum7 = vsubq_f64( qw_b7, qw_row_sum7 );
                    const float64x2_t qw_b_minus_sum8 = vsubq_f64( qw_b8, qw_row_sum8 );
                    const float64x2_t qw_x1_1         = vmulq_f64( qw_b_minus_sum1 , qw_Dinv1 );
                    const float64x2_t qw_x1_2         = vmulq_f64( qw_b_minus_sum2 , qw_Dinv2 );
                    const float64x2_t qw_x1_3         = vmulq_f64( qw_b_minus_sum3 , qw_Dinv3 );
                    const float64x2_t qw_x1_4         = vmulq_f64( qw_b_minus_sum4 , qw_Dinv4 );
                    const float64x2_t qw_x1_5         = vmulq_f64( qw_b_minus_sum5 , qw_Dinv5 );
                    const float64x2_t qw_x1_6         = vmulq_f64( qw_b_minus_sum6 , qw_Dinv6 );
                    const float64x2_t qw_x1_7         = vmulq_f64( qw_b_minus_sum7 , qw_Dinv7 );
                    const float64x2_t qw_x1_8         = vmulq_f64( qw_b_minus_sum8 , qw_Dinv8 );
                    memcpy( &(this->m_x1[i   ]), &qw_x1_1, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+ 2]), &qw_x1_2, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+ 4]), &qw_x1_3, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+ 6]), &qw_x1_4, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+ 8]), &qw_x1_5, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+10]), &qw_x1_6, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+12]), &qw_x1_7, sizeof(double)*2 );
                    memcpy( &(this->m_x1[i+14]), &qw_x1_8, sizeof(double)*2 );
                }
            }
        }
    }

    virtual void calcX2ColMajor( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            if ( m_factor_loop_unrolling == 1 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 4 ) {

                    float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const float       col_v   = this->m_x1[j];
                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float32x4_t qw_col  = { col_v, col_v, col_v, col_v };
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );

                        qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                    }

                    const float32x4_t qw_b1           = vld1q_f32( &(this->m_b[i  ]) );
                    const float32x4_t qw_Dinv1        = vld1q_f32( &(this->m_Dinv[i  ]) );
                    const float32x4_t qw_b_minus_sum1 = vsubq_f32( qw_b1, qw_row_sum1 );
                    const float32x4_t qw_x2_1         = vmulq_f32( qw_b_minus_sum1 , qw_Dinv1 );
                    memcpy( &(this->m_x2[i  ]), &qw_x2_1, sizeof(float)*4 );
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 8 ) {

                    float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const float       col_v   = this->m_x1[j];
                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ i + 4 + this->m_dim * j ]) );
                        const float32x4_t qw_col  = { col_v, col_v, col_v, col_v };
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );

                        qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                    }

                    const float32x4_t qw_b1           = vld1q_f32( &(this->m_b[i  ]) );
                    const float32x4_t qw_b2           = vld1q_f32( &(this->m_b[i+4]) );
                    const float32x4_t qw_Dinv1        = vld1q_f32( &(this->m_Dinv[i  ]) );
                    const float32x4_t qw_Dinv2        = vld1q_f32( &(this->m_Dinv[i+4]) );
                    const float32x4_t qw_b_minus_sum1 = vsubq_f32( qw_b1, qw_row_sum1 );
                    const float32x4_t qw_b_minus_sum2 = vsubq_f32( qw_b2, qw_row_sum2 );
                    const float32x4_t qw_x2_1         = vmulq_f32( qw_b_minus_sum1 , qw_Dinv1 );
                    const float32x4_t qw_x2_2         = vmulq_f32( qw_b_minus_sum2 , qw_Dinv2 );
                    memcpy( &(this->m_x2[i  ]), &qw_x2_1, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+4]), &qw_x2_2, sizeof(float)*4 );
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 16 ) {

                    float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum4 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const float       col_v   = this->m_x1[j];
                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ i      + this->m_dim * j ]) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ i +  4 + this->m_dim * j ]) );
                        const float32x4_t qw_mat3 = vld1q_f32( &(this->m_A[ i +  8 + this->m_dim * j ]) );
                        const float32x4_t qw_mat4 = vld1q_f32( &(this->m_A[ i + 12 + this->m_dim * j ]) );
                        const float32x4_t qw_col  = { col_v, col_v, col_v, col_v };
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );
                        const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col );
                        const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col );

                        qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                        qw_row_sum3 = vaddq_f32( qw_mc3, qw_row_sum3 );
                        qw_row_sum4 = vaddq_f32( qw_mc4, qw_row_sum4 );
                    }

                    const float32x4_t qw_b1           = vld1q_f32( &(this->m_b[i   ]) );
                    const float32x4_t qw_b2           = vld1q_f32( &(this->m_b[i+ 4]) );
                    const float32x4_t qw_b3           = vld1q_f32( &(this->m_b[i+ 8]) );
                    const float32x4_t qw_b4           = vld1q_f32( &(this->m_b[i+12]) );
                    const float32x4_t qw_Dinv1        = vld1q_f32( &(this->m_Dinv[i   ]) );
                    const float32x4_t qw_Dinv2        = vld1q_f32( &(this->m_Dinv[i+ 4]) );
                    const float32x4_t qw_Dinv3        = vld1q_f32( &(this->m_Dinv[i+ 8]) );
                    const float32x4_t qw_Dinv4        = vld1q_f32( &(this->m_Dinv[i+12]) );
                    const float32x4_t qw_b_minus_sum1 = vsubq_f32( qw_b1, qw_row_sum1 );
                    const float32x4_t qw_b_minus_sum2 = vsubq_f32( qw_b2, qw_row_sum2 );
                    const float32x4_t qw_b_minus_sum3 = vsubq_f32( qw_b3, qw_row_sum3 );
                    const float32x4_t qw_b_minus_sum4 = vsubq_f32( qw_b4, qw_row_sum4 );
                    const float32x4_t qw_x2_1         = vmulq_f32( qw_b_minus_sum1 , qw_Dinv1 );
                    const float32x4_t qw_x2_2         = vmulq_f32( qw_b_minus_sum2 , qw_Dinv2 );
                    const float32x4_t qw_x2_3         = vmulq_f32( qw_b_minus_sum3 , qw_Dinv3 );
                    const float32x4_t qw_x2_4         = vmulq_f32( qw_b_minus_sum4 , qw_Dinv4 );
                    memcpy( &(this->m_x2[i   ]), &qw_x2_1, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+ 4]), &qw_x2_2, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+ 8]), &qw_x2_3, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+12]), &qw_x2_4, sizeof(float)*4 );
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 32 ) {

                    float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum4 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum5 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum6 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum7 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_row_sum8 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const float       col_v   = this->m_x1[j];
                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ i      + this->m_dim * j ]) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ i +  4 + this->m_dim * j ]) );
                        const float32x4_t qw_mat3 = vld1q_f32( &(this->m_A[ i +  8 + this->m_dim * j ]) );
                        const float32x4_t qw_mat4 = vld1q_f32( &(this->m_A[ i + 12 + this->m_dim * j ]) );
                        const float32x4_t qw_mat5 = vld1q_f32( &(this->m_A[ i + 16 + this->m_dim * j ]) );
                        const float32x4_t qw_mat6 = vld1q_f32( &(this->m_A[ i + 20 + this->m_dim * j ]) );
                        const float32x4_t qw_mat7 = vld1q_f32( &(this->m_A[ i + 24 + this->m_dim * j ]) );
                        const float32x4_t qw_mat8 = vld1q_f32( &(this->m_A[ i + 28 + this->m_dim * j ]) );
                        const float32x4_t qw_col  = { col_v, col_v, col_v, col_v };
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );
                        const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col );
                        const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col );
                        const float32x4_t qw_mc5  = vmulq_f32( qw_mat5, qw_col );
                        const float32x4_t qw_mc6  = vmulq_f32( qw_mat6, qw_col );
                        const float32x4_t qw_mc7  = vmulq_f32( qw_mat7, qw_col );
                        const float32x4_t qw_mc8  = vmulq_f32( qw_mat8, qw_col );

                        qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                        qw_row_sum3 = vaddq_f32( qw_mc3, qw_row_sum3 );
                        qw_row_sum4 = vaddq_f32( qw_mc4, qw_row_sum4 );
                        qw_row_sum5 = vaddq_f32( qw_mc5, qw_row_sum5 );
                        qw_row_sum6 = vaddq_f32( qw_mc6, qw_row_sum6 );
                        qw_row_sum7 = vaddq_f32( qw_mc7, qw_row_sum7 );
                        qw_row_sum8 = vaddq_f32( qw_mc8, qw_row_sum8 );
                    }

                    const float32x4_t qw_b1           = vld1q_f32( &(this->m_b[i   ]) );
                    const float32x4_t qw_b2           = vld1q_f32( &(this->m_b[i+ 4]) );
                    const float32x4_t qw_b3           = vld1q_f32( &(this->m_b[i+ 8]) );
                    const float32x4_t qw_b4           = vld1q_f32( &(this->m_b[i+12]) );
                    const float32x4_t qw_b5           = vld1q_f32( &(this->m_b[i+16]) );
                    const float32x4_t qw_b6           = vld1q_f32( &(this->m_b[i+20]) );
                    const float32x4_t qw_b7           = vld1q_f32( &(this->m_b[i+24]) );
                    const float32x4_t qw_b8           = vld1q_f32( &(this->m_b[i+28]) );
                    const float32x4_t qw_Dinv1        = vld1q_f32( &(this->m_Dinv[i   ]) );
                    const float32x4_t qw_Dinv2        = vld1q_f32( &(this->m_Dinv[i+ 4]) );
                    const float32x4_t qw_Dinv3        = vld1q_f32( &(this->m_Dinv[i+ 8]) );
                    const float32x4_t qw_Dinv4        = vld1q_f32( &(this->m_Dinv[i+12]) );
                    const float32x4_t qw_Dinv5        = vld1q_f32( &(this->m_Dinv[i+16]) );
                    const float32x4_t qw_Dinv6        = vld1q_f32( &(this->m_Dinv[i+20]) );
                    const float32x4_t qw_Dinv7        = vld1q_f32( &(this->m_Dinv[i+24]) );
                    const float32x4_t qw_Dinv8        = vld1q_f32( &(this->m_Dinv[i+28]) );
                    const float32x4_t qw_b_minus_sum1 = vsubq_f32( qw_b1, qw_row_sum1 );
                    const float32x4_t qw_b_minus_sum2 = vsubq_f32( qw_b2, qw_row_sum2 );
                    const float32x4_t qw_b_minus_sum3 = vsubq_f32( qw_b3, qw_row_sum3 );
                    const float32x4_t qw_b_minus_sum4 = vsubq_f32( qw_b4, qw_row_sum4 );
                    const float32x4_t qw_b_minus_sum5 = vsubq_f32( qw_b5, qw_row_sum5 );
                    const float32x4_t qw_b_minus_sum6 = vsubq_f32( qw_b6, qw_row_sum6 );
                    const float32x4_t qw_b_minus_sum7 = vsubq_f32( qw_b7, qw_row_sum7 );
                    const float32x4_t qw_b_minus_sum8 = vsubq_f32( qw_b8, qw_row_sum8 );
                    const float32x4_t qw_x2_1         = vmulq_f32( qw_b_minus_sum1 , qw_Dinv1 );
                    const float32x4_t qw_x2_2         = vmulq_f32( qw_b_minus_sum2 , qw_Dinv2 );
                    const float32x4_t qw_x2_3         = vmulq_f32( qw_b_minus_sum3 , qw_Dinv3 );
                    const float32x4_t qw_x2_4         = vmulq_f32( qw_b_minus_sum4 , qw_Dinv4 );
                    const float32x4_t qw_x2_5         = vmulq_f32( qw_b_minus_sum5 , qw_Dinv5 );
                    const float32x4_t qw_x2_6         = vmulq_f32( qw_b_minus_sum6 , qw_Dinv6 );
                    const float32x4_t qw_x2_7         = vmulq_f32( qw_b_minus_sum7 , qw_Dinv7 );
                    const float32x4_t qw_x2_8         = vmulq_f32( qw_b_minus_sum8 , qw_Dinv8 );

                    memcpy( &(this->m_x2[i   ]), &qw_x2_1, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+ 4]), &qw_x2_2, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+ 8]), &qw_x2_3, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+12]), &qw_x2_4, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+16]), &qw_x2_5, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+20]), &qw_x2_6, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+24]), &qw_x2_7, sizeof(float)*4 );
                    memcpy( &(this->m_x2[i+28]), &qw_x2_8, sizeof(float)*4 );
                }
            }
        }
        else {
            if ( m_factor_loop_unrolling == 1 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 2 ) {

                    float64x2_t qw_row_sum1 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const double      col_v   = this->m_x1[j];
                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float64x2_t qw_col  = { col_v, col_v };
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );

                        qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                    }

                    const float64x2_t qw_b1           = vld1q_f64( &(this->m_b[i  ]) );
                    const float64x2_t qw_Dinv1        = vld1q_f64( &(this->m_Dinv[i  ]) );
                    const float64x2_t qw_b_minus_sum1 = vsubq_f64( qw_b1, qw_row_sum1 );
                    const float64x2_t qw_x2_1         = vmulq_f64( qw_b_minus_sum1 , qw_Dinv1 );
                    memcpy( &(this->m_x2[i  ]), &qw_x2_1, sizeof(double)*2 );
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 4 ) {

                    float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum2 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const double      col_v   = this->m_x1[j];
                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ i + 2 + this->m_dim * j ]) );
                        const float64x2_t qw_col  = { col_v, col_v };
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );

                        qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                    }

                    const float64x2_t qw_b1           = vld1q_f64( &(this->m_b[i  ]) );
                    const float64x2_t qw_b2           = vld1q_f64( &(this->m_b[i+2]) );
                    const float64x2_t qw_Dinv1        = vld1q_f64( &(this->m_Dinv[i  ]) );
                    const float64x2_t qw_Dinv2        = vld1q_f64( &(this->m_Dinv[i+2]) );
                    const float64x2_t qw_b_minus_sum1 = vsubq_f64( qw_b1, qw_row_sum1 );
                    const float64x2_t qw_b_minus_sum2 = vsubq_f64( qw_b2, qw_row_sum2 );
                    const float64x2_t qw_x2_1         = vmulq_f64( qw_b_minus_sum1 , qw_Dinv1 );
                    const float64x2_t qw_x2_2         = vmulq_f64( qw_b_minus_sum2 , qw_Dinv2 );
                    memcpy( &(this->m_x2[i  ]), &qw_x2_1, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+2]), &qw_x2_2, sizeof(double)*2 );
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 8 ) {

                    float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum2 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum3 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum4 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const double      col_v   = this->m_x1[j];
                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ i     + this->m_dim * j ]) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ i + 2 + this->m_dim * j ]) );
                        const float64x2_t qw_mat3 = vld1q_f64( &(this->m_A[ i + 4 + this->m_dim * j ]) );
                        const float64x2_t qw_mat4 = vld1q_f64( &(this->m_A[ i + 6 + this->m_dim * j ]) );
                        const float64x2_t qw_col  = { col_v, col_v };
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );
                        const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col );
                        const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col );

                        qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                        qw_row_sum3 = vaddq_f64( qw_mc3, qw_row_sum3 );
                        qw_row_sum4 = vaddq_f64( qw_mc4, qw_row_sum4 );
                    }

                    const float64x2_t qw_b1           = vld1q_f64( &(this->m_b[i  ]) );
                    const float64x2_t qw_b2           = vld1q_f64( &(this->m_b[i+2]) );
                    const float64x2_t qw_b3           = vld1q_f64( &(this->m_b[i+4]) );
                    const float64x2_t qw_b4           = vld1q_f64( &(this->m_b[i+6]) );
                    const float64x2_t qw_Dinv1        = vld1q_f64( &(this->m_Dinv[i  ]) );
                    const float64x2_t qw_Dinv2        = vld1q_f64( &(this->m_Dinv[i+2]) );
                    const float64x2_t qw_Dinv3        = vld1q_f64( &(this->m_Dinv[i+4]) );
                    const float64x2_t qw_Dinv4        = vld1q_f64( &(this->m_Dinv[i+6]) );
                    const float64x2_t qw_b_minus_sum1 = vsubq_f64( qw_b1, qw_row_sum1 );
                    const float64x2_t qw_b_minus_sum2 = vsubq_f64( qw_b2, qw_row_sum2 );
                    const float64x2_t qw_b_minus_sum3 = vsubq_f64( qw_b3, qw_row_sum3 );
                    const float64x2_t qw_b_minus_sum4 = vsubq_f64( qw_b4, qw_row_sum4 );
                    const float64x2_t qw_x2_1         = vmulq_f64( qw_b_minus_sum1 , qw_Dinv1 );
                    const float64x2_t qw_x2_2         = vmulq_f64( qw_b_minus_sum2 , qw_Dinv2 );
                    const float64x2_t qw_x2_3         = vmulq_f64( qw_b_minus_sum3 , qw_Dinv3 );
                    const float64x2_t qw_x2_4         = vmulq_f64( qw_b_minus_sum4 , qw_Dinv4 );
                    memcpy( &(this->m_x2[i  ]), &qw_x2_1, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+2]), &qw_x2_2, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+4]), &qw_x2_3, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+6]), &qw_x2_4, sizeof(double)*2 );
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for ( int i = row_begin; i < row_end_past_one; i += 16 ) {

                    float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum2 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum3 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum4 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum5 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum6 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum7 = { 0.0, 0.0 };
                    float64x2_t qw_row_sum8 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j++ ) {

                        const double      col_v   = this->m_x1[j];
                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ i      + this->m_dim * j ]) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ i +  2 + this->m_dim * j ]) );
                        const float64x2_t qw_mat3 = vld1q_f64( &(this->m_A[ i +  4 + this->m_dim * j ]) );
                        const float64x2_t qw_mat4 = vld1q_f64( &(this->m_A[ i +  6 + this->m_dim * j ]) );
                        const float64x2_t qw_mat5 = vld1q_f64( &(this->m_A[ i +  8 + this->m_dim * j ]) );
                        const float64x2_t qw_mat6 = vld1q_f64( &(this->m_A[ i + 10 + this->m_dim * j ]) );
                        const float64x2_t qw_mat7 = vld1q_f64( &(this->m_A[ i + 12 + this->m_dim * j ]) );
                        const float64x2_t qw_mat8 = vld1q_f64( &(this->m_A[ i + 14 + this->m_dim * j ]) );
                        const float64x2_t qw_col  = { col_v, col_v };
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );
                        const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col );
                        const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col );
                        const float64x2_t qw_mc5  = vmulq_f64( qw_mat5, qw_col );
                        const float64x2_t qw_mc6  = vmulq_f64( qw_mat6, qw_col );
                        const float64x2_t qw_mc7  = vmulq_f64( qw_mat7, qw_col );
                        const float64x2_t qw_mc8  = vmulq_f64( qw_mat8, qw_col );

                        qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                        qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                        qw_row_sum3 = vaddq_f64( qw_mc3, qw_row_sum3 );
                        qw_row_sum4 = vaddq_f64( qw_mc4, qw_row_sum4 );
                        qw_row_sum5 = vaddq_f64( qw_mc5, qw_row_sum5 );
                        qw_row_sum6 = vaddq_f64( qw_mc6, qw_row_sum6 );
                        qw_row_sum7 = vaddq_f64( qw_mc7, qw_row_sum7 );
                        qw_row_sum8 = vaddq_f64( qw_mc8, qw_row_sum8 );
                    }

                    const float64x2_t qw_b1           = vld1q_f64( &(this->m_b[i   ]) );
                    const float64x2_t qw_b2           = vld1q_f64( &(this->m_b[i+ 2]) );
                    const float64x2_t qw_b3           = vld1q_f64( &(this->m_b[i+ 4]) );
                    const float64x2_t qw_b4           = vld1q_f64( &(this->m_b[i+ 6]) );
                    const float64x2_t qw_b5           = vld1q_f64( &(this->m_b[i+ 8]) );
                    const float64x2_t qw_b6           = vld1q_f64( &(this->m_b[i+10]) );
                    const float64x2_t qw_b7           = vld1q_f64( &(this->m_b[i+12]) );
                    const float64x2_t qw_b8           = vld1q_f64( &(this->m_b[i+14]) );
                    const float64x2_t qw_Dinv1        = vld1q_f64( &(this->m_Dinv[i   ]) );
                    const float64x2_t qw_Dinv2        = vld1q_f64( &(this->m_Dinv[i+ 2]) );
                    const float64x2_t qw_Dinv3        = vld1q_f64( &(this->m_Dinv[i+ 4]) );
                    const float64x2_t qw_Dinv4        = vld1q_f64( &(this->m_Dinv[i+ 6]) );
                    const float64x2_t qw_Dinv5        = vld1q_f64( &(this->m_Dinv[i+ 8]) );
                    const float64x2_t qw_Dinv6        = vld1q_f64( &(this->m_Dinv[i+10]) );
                    const float64x2_t qw_Dinv7        = vld1q_f64( &(this->m_Dinv[i+12]) );
                    const float64x2_t qw_Dinv8        = vld1q_f64( &(this->m_Dinv[i+14]) );
                    const float64x2_t qw_b_minus_sum1 = vsubq_f64( qw_b1, qw_row_sum1 );
                    const float64x2_t qw_b_minus_sum2 = vsubq_f64( qw_b2, qw_row_sum2 );
                    const float64x2_t qw_b_minus_sum3 = vsubq_f64( qw_b3, qw_row_sum3 );
                    const float64x2_t qw_b_minus_sum4 = vsubq_f64( qw_b4, qw_row_sum4 );
                    const float64x2_t qw_b_minus_sum5 = vsubq_f64( qw_b5, qw_row_sum5 );
                    const float64x2_t qw_b_minus_sum6 = vsubq_f64( qw_b6, qw_row_sum6 );
                    const float64x2_t qw_b_minus_sum7 = vsubq_f64( qw_b7, qw_row_sum7 );
                    const float64x2_t qw_b_minus_sum8 = vsubq_f64( qw_b8, qw_row_sum8 );
                    const float64x2_t qw_x2_1         = vmulq_f64( qw_b_minus_sum1 , qw_Dinv1 );
                    const float64x2_t qw_x2_2         = vmulq_f64( qw_b_minus_sum2 , qw_Dinv2 );
                    const float64x2_t qw_x2_3         = vmulq_f64( qw_b_minus_sum3 , qw_Dinv3 );
                    const float64x2_t qw_x2_4         = vmulq_f64( qw_b_minus_sum4 , qw_Dinv4 );
                    const float64x2_t qw_x2_5         = vmulq_f64( qw_b_minus_sum5 , qw_Dinv5 );
                    const float64x2_t qw_x2_6         = vmulq_f64( qw_b_minus_sum6 , qw_Dinv6 );
                    const float64x2_t qw_x2_7         = vmulq_f64( qw_b_minus_sum7 , qw_Dinv7 );
                    const float64x2_t qw_x2_8         = vmulq_f64( qw_b_minus_sum8 , qw_Dinv8 );
                    memcpy( &(this->m_x2[i   ]), &qw_x2_1, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+ 2]), &qw_x2_2, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+ 4]), &qw_x2_3, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+ 6]), &qw_x2_4, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+ 8]), &qw_x2_5, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+10]), &qw_x2_6, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+12]), &qw_x2_7, sizeof(double)*2 );
                    memcpy( &(this->m_x2[i+14]), &qw_x2_8, sizeof(double)*2 );
                }
            }
        }
    }

    virtual void calcX1RowMajor( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            if ( m_factor_loop_unrolling == 1 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=4 ) {

                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float32x4_t qw_col1 = vld1q_f32( &(this->m_x2[ j     ] ) );
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                        qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                    }
                    const float sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3];
                    this->m_x1[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=8 ) {

                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 4 ] ) );
                        const float32x4_t qw_col1 = vld1q_f32( &(this->m_x2[ j     ] ) );
                        const float32x4_t qw_col2 = vld1q_f32( &(this->m_x2[ j + 4 ] ) );
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                        qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                    }
                    const float sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                      + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3];
                    this->m_x1[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum4 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=16 ) {

                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ this->m_dim * i + j      ] ) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ this->m_dim * i + j +  4 ] ) );
                        const float32x4_t qw_mat3 = vld1q_f32( &(this->m_A[ this->m_dim * i + j +  8 ] ) );
                        const float32x4_t qw_mat4 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 12 ] ) );
                        const float32x4_t qw_col1 = vld1q_f32( &(this->m_x2[ j      ] ) );
                        const float32x4_t qw_col2 = vld1q_f32( &(this->m_x2[ j +  4 ] ) );
                        const float32x4_t qw_col3 = vld1q_f32( &(this->m_x2[ j +  8 ] ) );
                        const float32x4_t qw_col4 = vld1q_f32( &(this->m_x2[ j + 12 ] ) );
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                        const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col3 );
                        const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col4 );
                        qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                        qw_lanewise_sum3 = vaddq_f32( qw_mc3, qw_lanewise_sum3 );
                        qw_lanewise_sum4 = vaddq_f32( qw_mc4, qw_lanewise_sum4 );
                    }
                    const float sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                      + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3]
                                      + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum3[2] + qw_lanewise_sum3[3]
                                      + qw_lanewise_sum4[0] + qw_lanewise_sum4[1] + qw_lanewise_sum4[2] + qw_lanewise_sum4[3];
                    this->m_x1[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum4 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum5 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum6 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum7 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum8 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=32 ) {

                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ this->m_dim * i + j      ] ) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ this->m_dim * i + j +  4 ] ) );
                        const float32x4_t qw_mat3 = vld1q_f32( &(this->m_A[ this->m_dim * i + j +  8 ] ) );
                        const float32x4_t qw_mat4 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 12 ] ) );
                        const float32x4_t qw_mat5 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 16 ] ) );
                        const float32x4_t qw_mat6 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 20 ] ) );
                        const float32x4_t qw_mat7 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 24 ] ) );
                        const float32x4_t qw_mat8 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 28 ] ) );
                        const float32x4_t qw_col1 = vld1q_f32( &(this->m_x2[ j      ] ) );
                        const float32x4_t qw_col2 = vld1q_f32( &(this->m_x2[ j +  4 ] ) );
                        const float32x4_t qw_col3 = vld1q_f32( &(this->m_x2[ j +  8 ] ) );
                        const float32x4_t qw_col4 = vld1q_f32( &(this->m_x2[ j + 12 ] ) );
                        const float32x4_t qw_col5 = vld1q_f32( &(this->m_x2[ j + 16 ] ) );
                        const float32x4_t qw_col6 = vld1q_f32( &(this->m_x2[ j + 20 ] ) );
                        const float32x4_t qw_col7 = vld1q_f32( &(this->m_x2[ j + 24 ] ) );
                        const float32x4_t qw_col8 = vld1q_f32( &(this->m_x2[ j + 28 ] ) );
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                        const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col3 );
                        const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col4 );
                        const float32x4_t qw_mc5  = vmulq_f32( qw_mat5, qw_col5 );
                        const float32x4_t qw_mc6  = vmulq_f32( qw_mat6, qw_col6 );
                        const float32x4_t qw_mc7  = vmulq_f32( qw_mat7, qw_col7 );
                        const float32x4_t qw_mc8  = vmulq_f32( qw_mat8, qw_col8 );
                        qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                        qw_lanewise_sum3 = vaddq_f32( qw_mc3, qw_lanewise_sum3 );
                        qw_lanewise_sum4 = vaddq_f32( qw_mc4, qw_lanewise_sum4 );
                        qw_lanewise_sum5 = vaddq_f32( qw_mc5, qw_lanewise_sum5 );
                        qw_lanewise_sum6 = vaddq_f32( qw_mc6, qw_lanewise_sum6 );
                        qw_lanewise_sum7 = vaddq_f32( qw_mc7, qw_lanewise_sum7 );
                        qw_lanewise_sum8 = vaddq_f32( qw_mc8, qw_lanewise_sum8 );
                    }
                    const float sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                      + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3]
                                      + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum3[2] + qw_lanewise_sum3[3]
                                      + qw_lanewise_sum4[0] + qw_lanewise_sum4[1] + qw_lanewise_sum4[2] + qw_lanewise_sum4[3]
                                      + qw_lanewise_sum5[0] + qw_lanewise_sum5[1] + qw_lanewise_sum5[2] + qw_lanewise_sum5[3]
                                      + qw_lanewise_sum6[0] + qw_lanewise_sum6[1] + qw_lanewise_sum6[2] + qw_lanewise_sum6[3]
                                      + qw_lanewise_sum7[0] + qw_lanewise_sum7[1] + qw_lanewise_sum7[2] + qw_lanewise_sum7[3]
                                      + qw_lanewise_sum8[0] + qw_lanewise_sum8[1] + qw_lanewise_sum8[2] + qw_lanewise_sum8[3];
                    this->m_x1[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }

        }
        else {
            if ( m_factor_loop_unrolling == 1 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=2 ) {

                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float64x2_t qw_col1 = vld1q_f64( &(this->m_x2[ j     ] ) );
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                        qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                    }
                    const double sum = qw_lanewise_sum1[0] + qw_lanewise_sum1[1];
                             
                    this->m_x1[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=4 ) {

                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 2 ] ) );
                        const float64x2_t qw_col1 = vld1q_f64( &(this->m_x2[ j     ] ) );
                        const float64x2_t qw_col2 = vld1q_f64( &(this->m_x2[ j + 2 ] ) );
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                        qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                    }
                    const double sum = qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1];
                             
                    this->m_x1[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum3 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum4 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=8 ) {

                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 2 ] ) );
                        const float64x2_t qw_mat3 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 4 ] ) );
                        const float64x2_t qw_mat4 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 6 ] ) );
                        const float64x2_t qw_col1 = vld1q_f64( &(this->m_x2[ j     ] ) );
                        const float64x2_t qw_col2 = vld1q_f64( &(this->m_x2[ j + 2 ] ) );
                        const float64x2_t qw_col3 = vld1q_f64( &(this->m_x2[ j + 4 ] ) );
                        const float64x2_t qw_col4 = vld1q_f64( &(this->m_x2[ j + 6 ] ) );
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                        const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col3 );
                        const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col4 );
                        qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                        qw_lanewise_sum3 = vaddq_f64( qw_mc3, qw_lanewise_sum3 );
                        qw_lanewise_sum4 = vaddq_f64( qw_mc4, qw_lanewise_sum4 );
                    }
                    const double sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1]
                                       + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum4[0] + qw_lanewise_sum4[1];
                             
                    this->m_x1[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum3 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum4 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum5 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum6 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum7 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum8 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=16 ) {

                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ this->m_dim * i + j      ] ) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ this->m_dim * i + j +  2 ] ) );
                        const float64x2_t qw_mat3 = vld1q_f64( &(this->m_A[ this->m_dim * i + j +  4 ] ) );
                        const float64x2_t qw_mat4 = vld1q_f64( &(this->m_A[ this->m_dim * i + j +  6 ] ) );
                        const float64x2_t qw_mat5 = vld1q_f64( &(this->m_A[ this->m_dim * i + j +  8 ] ) );
                        const float64x2_t qw_mat6 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 10 ] ) );
                        const float64x2_t qw_mat7 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 12 ] ) );
                        const float64x2_t qw_mat8 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 14 ] ) );
                        const float64x2_t qw_col1 = vld1q_f64( &(this->m_x2[ j      ] ) );
                        const float64x2_t qw_col2 = vld1q_f64( &(this->m_x2[ j +  2 ] ) );
                        const float64x2_t qw_col3 = vld1q_f64( &(this->m_x2[ j +  4 ] ) );
                        const float64x2_t qw_col4 = vld1q_f64( &(this->m_x2[ j +  6 ] ) );
                        const float64x2_t qw_col5 = vld1q_f64( &(this->m_x2[ j +  8 ] ) );
                        const float64x2_t qw_col6 = vld1q_f64( &(this->m_x2[ j + 10 ] ) );
                        const float64x2_t qw_col7 = vld1q_f64( &(this->m_x2[ j + 12 ] ) );
                        const float64x2_t qw_col8 = vld1q_f64( &(this->m_x2[ j + 14 ] ) );
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                        const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col3 );
                        const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col4 );
                        const float64x2_t qw_mc5  = vmulq_f64( qw_mat5, qw_col5 );
                        const float64x2_t qw_mc6  = vmulq_f64( qw_mat6, qw_col6 );
                        const float64x2_t qw_mc7  = vmulq_f64( qw_mat7, qw_col7 );
                        const float64x2_t qw_mc8  = vmulq_f64( qw_mat8, qw_col8 );
                        qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                        qw_lanewise_sum3 = vaddq_f64( qw_mc3, qw_lanewise_sum3 );
                        qw_lanewise_sum4 = vaddq_f64( qw_mc4, qw_lanewise_sum4 );
                        qw_lanewise_sum5 = vaddq_f64( qw_mc5, qw_lanewise_sum5 );
                        qw_lanewise_sum6 = vaddq_f64( qw_mc6, qw_lanewise_sum6 );
                        qw_lanewise_sum7 = vaddq_f64( qw_mc7, qw_lanewise_sum7 );
                        qw_lanewise_sum8 = vaddq_f64( qw_mc8, qw_lanewise_sum8 );
                    }
                    const double sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1]
                                       + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum4[0] + qw_lanewise_sum4[1]
                                       + qw_lanewise_sum5[0] + qw_lanewise_sum5[1] + qw_lanewise_sum6[0] + qw_lanewise_sum6[1]
                                       + qw_lanewise_sum7[0] + qw_lanewise_sum7[1] + qw_lanewise_sum8[0] + qw_lanewise_sum8[1];
                             
                    this->m_x1[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
        }
    }

    virtual void calcX2RowMajor( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            if ( m_factor_loop_unrolling == 1 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=4 ) {

                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float32x4_t qw_col1 = vld1q_f32( &(this->m_x1[ j     ] ) );
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                        qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                    }
                    const float sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3];
                    this->m_x2[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=8 ) {

                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 4 ] ) );
                        const float32x4_t qw_col1 = vld1q_f32( &(this->m_x1[ j     ] ) );
                        const float32x4_t qw_col2 = vld1q_f32( &(this->m_x1[ j + 4 ] ) );
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                        qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                    }
                    const float sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                      + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3];
                    this->m_x2[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum4 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=16 ) {

                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ this->m_dim * i + j      ] ) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ this->m_dim * i + j +  4 ] ) );
                        const float32x4_t qw_mat3 = vld1q_f32( &(this->m_A[ this->m_dim * i + j +  8 ] ) );
                        const float32x4_t qw_mat4 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 12 ] ) );
                        const float32x4_t qw_col1 = vld1q_f32( &(this->m_x1[ j      ] ) );
                        const float32x4_t qw_col2 = vld1q_f32( &(this->m_x1[ j +  4 ] ) );
                        const float32x4_t qw_col3 = vld1q_f32( &(this->m_x1[ j +  8 ] ) );
                        const float32x4_t qw_col4 = vld1q_f32( &(this->m_x1[ j + 12 ] ) );
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                        const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col3 );
                        const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col4 );
                        qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                        qw_lanewise_sum3 = vaddq_f32( qw_mc3, qw_lanewise_sum3 );
                        qw_lanewise_sum4 = vaddq_f32( qw_mc4, qw_lanewise_sum4 );
                    }
                    const float sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                      + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3]
                                      + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum3[2] + qw_lanewise_sum3[3]
                                      + qw_lanewise_sum4[0] + qw_lanewise_sum4[1] + qw_lanewise_sum4[2] + qw_lanewise_sum4[3];
                    this->m_x2[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum4 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum5 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum6 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum7 = { 0.0, 0.0, 0.0, 0.0 };
                    float32x4_t qw_lanewise_sum8 = { 0.0, 0.0, 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=32 ) {

                        const float32x4_t qw_mat1 = vld1q_f32( &(this->m_A[ this->m_dim * i + j      ] ) );
                        const float32x4_t qw_mat2 = vld1q_f32( &(this->m_A[ this->m_dim * i + j +  4 ] ) );
                        const float32x4_t qw_mat3 = vld1q_f32( &(this->m_A[ this->m_dim * i + j +  8 ] ) );
                        const float32x4_t qw_mat4 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 12 ] ) );
                        const float32x4_t qw_mat5 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 16 ] ) );
                        const float32x4_t qw_mat6 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 20 ] ) );
                        const float32x4_t qw_mat7 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 24 ] ) );
                        const float32x4_t qw_mat8 = vld1q_f32( &(this->m_A[ this->m_dim * i + j + 28 ] ) );
                        const float32x4_t qw_col1 = vld1q_f32( &(this->m_x1[ j      ] ) );
                        const float32x4_t qw_col2 = vld1q_f32( &(this->m_x1[ j +  4 ] ) );
                        const float32x4_t qw_col3 = vld1q_f32( &(this->m_x1[ j +  8 ] ) );
                        const float32x4_t qw_col4 = vld1q_f32( &(this->m_x1[ j + 12 ] ) );
                        const float32x4_t qw_col5 = vld1q_f32( &(this->m_x1[ j + 16 ] ) );
                        const float32x4_t qw_col6 = vld1q_f32( &(this->m_x1[ j + 20 ] ) );
                        const float32x4_t qw_col7 = vld1q_f32( &(this->m_x1[ j + 24 ] ) );
                        const float32x4_t qw_col8 = vld1q_f32( &(this->m_x1[ j + 28 ] ) );
                        const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                        const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                        const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col3 );
                        const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col4 );
                        const float32x4_t qw_mc5  = vmulq_f32( qw_mat5, qw_col5 );
                        const float32x4_t qw_mc6  = vmulq_f32( qw_mat6, qw_col6 );
                        const float32x4_t qw_mc7  = vmulq_f32( qw_mat7, qw_col7 );
                        const float32x4_t qw_mc8  = vmulq_f32( qw_mat8, qw_col8 );
                        qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                        qw_lanewise_sum3 = vaddq_f32( qw_mc3, qw_lanewise_sum3 );
                        qw_lanewise_sum4 = vaddq_f32( qw_mc4, qw_lanewise_sum4 );
                        qw_lanewise_sum5 = vaddq_f32( qw_mc5, qw_lanewise_sum5 );
                        qw_lanewise_sum6 = vaddq_f32( qw_mc6, qw_lanewise_sum6 );
                        qw_lanewise_sum7 = vaddq_f32( qw_mc7, qw_lanewise_sum7 );
                        qw_lanewise_sum8 = vaddq_f32( qw_mc8, qw_lanewise_sum8 );
                    }
                    const float sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                      + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3]
                                      + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum3[2] + qw_lanewise_sum3[3]
                                      + qw_lanewise_sum4[0] + qw_lanewise_sum4[1] + qw_lanewise_sum4[2] + qw_lanewise_sum4[3]
                                      + qw_lanewise_sum5[0] + qw_lanewise_sum5[1] + qw_lanewise_sum5[2] + qw_lanewise_sum5[3]
                                      + qw_lanewise_sum6[0] + qw_lanewise_sum6[1] + qw_lanewise_sum6[2] + qw_lanewise_sum6[3]
                                      + qw_lanewise_sum7[0] + qw_lanewise_sum7[1] + qw_lanewise_sum7[2] + qw_lanewise_sum7[3]
                                      + qw_lanewise_sum8[0] + qw_lanewise_sum8[1] + qw_lanewise_sum8[2] + qw_lanewise_sum8[3];
                    this->m_x2[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }

        }
        else {
            if ( m_factor_loop_unrolling == 1 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=2 ) {

                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float64x2_t qw_col1 = vld1q_f64( &(this->m_x1[ j     ] ) );
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                        qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                    }
                    const double sum = qw_lanewise_sum1[0] + qw_lanewise_sum1[1];
                             
                    this->m_x2[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=4 ) {

                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 2 ] ) );
                        const float64x2_t qw_col1 = vld1q_f64( &(this->m_x1[ j     ] ) );
                        const float64x2_t qw_col2 = vld1q_f64( &(this->m_x1[ j + 2 ] ) );
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                        qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                    }
                    const double sum = qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1];
                             
                    this->m_x2[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum3 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum4 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=8 ) {

                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ this->m_dim * i + j     ] ) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 2 ] ) );
                        const float64x2_t qw_mat3 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 4 ] ) );
                        const float64x2_t qw_mat4 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 6 ] ) );
                        const float64x2_t qw_col1 = vld1q_f64( &(this->m_x1[ j     ] ) );
                        const float64x2_t qw_col2 = vld1q_f64( &(this->m_x1[ j + 2 ] ) );
                        const float64x2_t qw_col3 = vld1q_f64( &(this->m_x1[ j + 4 ] ) );
                        const float64x2_t qw_col4 = vld1q_f64( &(this->m_x1[ j + 6 ] ) );
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                        const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col3 );
                        const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col4 );
                        qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                        qw_lanewise_sum3 = vaddq_f64( qw_mc3, qw_lanewise_sum3 );
                        qw_lanewise_sum4 = vaddq_f64( qw_mc4, qw_lanewise_sum4 );
                    }
                    const double sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1]
                                       + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum4[0] + qw_lanewise_sum4[1];
                             
                    this->m_x2[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                for ( int i = row_begin; i < row_end_past_one; i++ ) {

                    float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum3 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum4 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum5 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum6 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum7 = { 0.0, 0.0 };
                    float64x2_t qw_lanewise_sum8 = { 0.0, 0.0 };

                    for ( int j = 0; j < this->m_dim; j+=16 ) {

                        const float64x2_t qw_mat1 = vld1q_f64( &(this->m_A[ this->m_dim * i + j      ] ) );
                        const float64x2_t qw_mat2 = vld1q_f64( &(this->m_A[ this->m_dim * i + j +  2 ] ) );
                        const float64x2_t qw_mat3 = vld1q_f64( &(this->m_A[ this->m_dim * i + j +  4 ] ) );
                        const float64x2_t qw_mat4 = vld1q_f64( &(this->m_A[ this->m_dim * i + j +  6 ] ) );
                        const float64x2_t qw_mat5 = vld1q_f64( &(this->m_A[ this->m_dim * i + j +  8 ] ) );
                        const float64x2_t qw_mat6 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 10 ] ) );
                        const float64x2_t qw_mat7 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 12 ] ) );
                        const float64x2_t qw_mat8 = vld1q_f64( &(this->m_A[ this->m_dim * i + j + 14 ] ) );
                        const float64x2_t qw_col1 = vld1q_f64( &(this->m_x1[ j      ] ) );
                        const float64x2_t qw_col2 = vld1q_f64( &(this->m_x1[ j +  2 ] ) );
                        const float64x2_t qw_col3 = vld1q_f64( &(this->m_x1[ j +  4 ] ) );
                        const float64x2_t qw_col4 = vld1q_f64( &(this->m_x1[ j +  6 ] ) );
                        const float64x2_t qw_col5 = vld1q_f64( &(this->m_x1[ j +  8 ] ) );
                        const float64x2_t qw_col6 = vld1q_f64( &(this->m_x1[ j + 10 ] ) );
                        const float64x2_t qw_col7 = vld1q_f64( &(this->m_x1[ j + 12 ] ) );
                        const float64x2_t qw_col8 = vld1q_f64( &(this->m_x1[ j + 14 ] ) );
                        const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                        const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                        const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col3 );
                        const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col4 );
                        const float64x2_t qw_mc5  = vmulq_f64( qw_mat5, qw_col5 );
                        const float64x2_t qw_mc6  = vmulq_f64( qw_mat6, qw_col6 );
                        const float64x2_t qw_mc7  = vmulq_f64( qw_mat7, qw_col7 );
                        const float64x2_t qw_mc8  = vmulq_f64( qw_mat8, qw_col8 );
                        qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                        qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                        qw_lanewise_sum3 = vaddq_f64( qw_mc3, qw_lanewise_sum3 );
                        qw_lanewise_sum4 = vaddq_f64( qw_mc4, qw_lanewise_sum4 );
                        qw_lanewise_sum5 = vaddq_f64( qw_mc5, qw_lanewise_sum5 );
                        qw_lanewise_sum6 = vaddq_f64( qw_mc6, qw_lanewise_sum6 );
                        qw_lanewise_sum7 = vaddq_f64( qw_mc7, qw_lanewise_sum7 );
                        qw_lanewise_sum8 = vaddq_f64( qw_mc8, qw_lanewise_sum8 );
                    }
                    const double sum =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1]
                                       + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum4[0] + qw_lanewise_sum4[1]
                                       + qw_lanewise_sum5[0] + qw_lanewise_sum5[1] + qw_lanewise_sum6[0] + qw_lanewise_sum6[1]
                                       + qw_lanewise_sum7[0] + qw_lanewise_sum7[1] + qw_lanewise_sum8[0] + qw_lanewise_sum8[1];
                             
                    this->m_x2[i] = (this->m_b[i] - sum ) * this->m_Dinv[i];
                }
            }
        }
    }


    virtual T getRmsX1X2Neon()
    {
        if constexpr ( is_same< float,T >::value ) {

            if ( m_factor_loop_unrolling == 1 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=4 ) {

                    const float32x4_t qw_x1_1  = vld1q_f32( &(this->m_x1[ i     ] ) );
                    const float32x4_t qw_x2_1  = vld1q_f32( &(this->m_x2[ i     ] ) );
                    const float32x4_t qw_diff1 = vsubq_f32( qw_x1_1, qw_x2_1 );
                    const float32x4_t qw_sq1   = vmulq_f32( qw_diff1,  qw_diff1 ); 
                    sum_sq_dist += ( qw_sq1[0] + qw_sq1[1] + qw_sq1[2] + qw_sq1[3] );
                }
                return sqrt( sum_sq_dist / this->m_dim );
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=8 ) {

                    const float32x4_t qw_x1_1  = vld1q_f32( &(this->m_x1[ i     ] ) );
                    const float32x4_t qw_x1_2  = vld1q_f32( &(this->m_x1[ i + 4 ] ) );
                    const float32x4_t qw_x2_1  = vld1q_f32( &(this->m_x2[ i     ] ) );
                    const float32x4_t qw_x2_2  = vld1q_f32( &(this->m_x2[ i + 4 ] ) );
                    const float32x4_t qw_diff1 = vsubq_f32( qw_x1_1, qw_x2_1 );
                    const float32x4_t qw_diff2 = vsubq_f32( qw_x1_2, qw_x2_2 );
                    const float32x4_t qw_sq1   = vmulq_f32( qw_diff1,  qw_diff1 ); 
                    const float32x4_t qw_sq2   = vmulq_f32( qw_diff2,  qw_diff2 ); 
                    sum_sq_dist += ( qw_sq1[0] + qw_sq1[1] + qw_sq1[2] + qw_sq1[3] + qw_sq2[0] + qw_sq2[1] + qw_sq2[2] + qw_sq2[3] );
                }
                return sqrt( sum_sq_dist / this->m_dim );
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=16 ) {

                    const float32x4_t qw_x1_1  = vld1q_f32( &(this->m_x1[ i      ] ) );
                    const float32x4_t qw_x1_2  = vld1q_f32( &(this->m_x1[ i +  4 ] ) );
                    const float32x4_t qw_x1_3  = vld1q_f32( &(this->m_x1[ i +  8 ] ) );
                    const float32x4_t qw_x1_4  = vld1q_f32( &(this->m_x1[ i + 12 ] ) );
                    const float32x4_t qw_x2_1  = vld1q_f32( &(this->m_x2[ i      ] ) );
                    const float32x4_t qw_x2_2  = vld1q_f32( &(this->m_x2[ i +  4 ] ) );
                    const float32x4_t qw_x2_3  = vld1q_f32( &(this->m_x2[ i +  8 ] ) );
                    const float32x4_t qw_x2_4  = vld1q_f32( &(this->m_x2[ i + 12 ] ) );
                    const float32x4_t qw_diff1 = vsubq_f32( qw_x1_1, qw_x2_1 );
                    const float32x4_t qw_diff2 = vsubq_f32( qw_x1_2, qw_x2_2 );
                    const float32x4_t qw_diff3 = vsubq_f32( qw_x1_3, qw_x2_3 );
                    const float32x4_t qw_diff4 = vsubq_f32( qw_x1_4, qw_x2_4 );
                    const float32x4_t qw_sq1   = vmulq_f32( qw_diff1,  qw_diff1 ); 
                    const float32x4_t qw_sq2   = vmulq_f32( qw_diff2,  qw_diff2 ); 
                    const float32x4_t qw_sq3   = vmulq_f32( qw_diff3,  qw_diff3 ); 
                    const float32x4_t qw_sq4   = vmulq_f32( qw_diff4,  qw_diff4 ); 
                    sum_sq_dist += (   qw_sq1[0] + qw_sq1[1] + qw_sq1[2] + qw_sq1[3] + qw_sq2[0] + qw_sq2[1] + qw_sq2[2] + qw_sq2[3] 
                                     + qw_sq3[0] + qw_sq3[1] + qw_sq3[2] + qw_sq3[3] + qw_sq4[0] + qw_sq4[1] + qw_sq4[2] + qw_sq4[3] );
                }
                return sqrt( sum_sq_dist / this->m_dim );
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=16 ) {

                    const float32x4_t qw_x1_1  = vld1q_f32( &(this->m_x1[ i      ] ) );
                    const float32x4_t qw_x1_2  = vld1q_f32( &(this->m_x1[ i +  4 ] ) );
                    const float32x4_t qw_x1_3  = vld1q_f32( &(this->m_x1[ i +  8 ] ) );
                    const float32x4_t qw_x1_4  = vld1q_f32( &(this->m_x1[ i + 12 ] ) );
                    const float32x4_t qw_x1_5  = vld1q_f32( &(this->m_x1[ i + 16 ] ) );
                    const float32x4_t qw_x1_6  = vld1q_f32( &(this->m_x1[ i + 20 ] ) );
                    const float32x4_t qw_x1_7  = vld1q_f32( &(this->m_x1[ i + 24 ] ) );
                    const float32x4_t qw_x1_8  = vld1q_f32( &(this->m_x1[ i + 28 ] ) );
                    const float32x4_t qw_x2_1  = vld1q_f32( &(this->m_x2[ i      ] ) );
                    const float32x4_t qw_x2_2  = vld1q_f32( &(this->m_x2[ i +  4 ] ) );
                    const float32x4_t qw_x2_3  = vld1q_f32( &(this->m_x2[ i +  8 ] ) );
                    const float32x4_t qw_x2_4  = vld1q_f32( &(this->m_x2[ i + 12 ] ) );
                    const float32x4_t qw_x2_5  = vld1q_f32( &(this->m_x2[ i + 16 ] ) );
                    const float32x4_t qw_x2_6  = vld1q_f32( &(this->m_x2[ i + 20 ] ) );
                    const float32x4_t qw_x2_7  = vld1q_f32( &(this->m_x2[ i + 24 ] ) );
                    const float32x4_t qw_x2_8  = vld1q_f32( &(this->m_x2[ i + 28 ] ) );
                    const float32x4_t qw_diff1 = vsubq_f32( qw_x1_1, qw_x2_1 );
                    const float32x4_t qw_diff2 = vsubq_f32( qw_x1_2, qw_x2_2 );
                    const float32x4_t qw_diff3 = vsubq_f32( qw_x1_3, qw_x2_3 );
                    const float32x4_t qw_diff4 = vsubq_f32( qw_x1_4, qw_x2_4 );
                    const float32x4_t qw_diff5 = vsubq_f32( qw_x1_5, qw_x2_5 );
                    const float32x4_t qw_diff6 = vsubq_f32( qw_x1_6, qw_x2_6 );
                    const float32x4_t qw_diff7 = vsubq_f32( qw_x1_7, qw_x2_7 );
                    const float32x4_t qw_diff8 = vsubq_f32( qw_x1_8, qw_x2_8 );
                    const float32x4_t qw_sq1   = vmulq_f32( qw_diff1,  qw_diff1 ); 
                    const float32x4_t qw_sq2   = vmulq_f32( qw_diff2,  qw_diff2 ); 
                    const float32x4_t qw_sq3   = vmulq_f32( qw_diff3,  qw_diff3 ); 
                    const float32x4_t qw_sq4   = vmulq_f32( qw_diff4,  qw_diff4 ); 
                    const float32x4_t qw_sq5   = vmulq_f32( qw_diff5,  qw_diff5 ); 
                    const float32x4_t qw_sq6   = vmulq_f32( qw_diff6,  qw_diff6 ); 
                    const float32x4_t qw_sq7   = vmulq_f32( qw_diff7,  qw_diff7 ); 
                    const float32x4_t qw_sq8   = vmulq_f32( qw_diff8,  qw_diff8 ); 
                    sum_sq_dist += (   qw_sq1[0] + qw_sq1[1] + qw_sq1[2] + qw_sq1[3] + qw_sq2[0] + qw_sq2[1] + qw_sq2[2] + qw_sq2[3] 
                                     + qw_sq3[0] + qw_sq3[1] + qw_sq3[2] + qw_sq3[3] + qw_sq4[0] + qw_sq4[1] + qw_sq4[2] + qw_sq4[3]
                                     + qw_sq5[0] + qw_sq5[1] + qw_sq5[2] + qw_sq5[3] + qw_sq6[0] + qw_sq6[1] + qw_sq6[2] + qw_sq6[3] 
                                     + qw_sq7[0] + qw_sq7[1] + qw_sq7[2] + qw_sq7[3] + qw_sq8[0] + qw_sq8[1] + qw_sq8[2] + qw_sq8[3] );
                }
                return sqrt( sum_sq_dist / this->m_dim );
            }
        }
        else {
            if ( m_factor_loop_unrolling == 1 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=4 ) {

                    const float64x2_t qw_x1_1  = vld1q_f64( &(this->m_x1[ i     ] ) );
                    const float64x2_t qw_x2_1  = vld1q_f64( &(this->m_x2[ i     ] ) );
                    const float64x2_t qw_diff1 = vsubq_f64( qw_x1_1, qw_x2_1 );
                    const float64x2_t qw_sq1   = vmulq_f64( qw_diff1,  qw_diff1 ); 
                    sum_sq_dist += ( qw_sq1[0] + qw_sq1[1] );
                }
                return sqrt( sum_sq_dist / this->m_dim );
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=4 ) {

                    const float64x2_t qw_x1_1  = vld1q_f64( &(this->m_x1[ i     ] ) );
                    const float64x2_t qw_x1_2  = vld1q_f64( &(this->m_x1[ i + 2 ] ) );
                    const float64x2_t qw_x2_1  = vld1q_f64( &(this->m_x2[ i     ] ) );
                    const float64x2_t qw_x2_2  = vld1q_f64( &(this->m_x2[ i + 2 ] ) );
                    const float64x2_t qw_diff1 = vsubq_f64( qw_x1_1, qw_x2_1 );
                    const float64x2_t qw_diff2 = vsubq_f64( qw_x1_2, qw_x2_2 );
                    const float64x2_t qw_sq1   = vmulq_f64( qw_diff1,  qw_diff1 ); 
                    const float64x2_t qw_sq2   = vmulq_f64( qw_diff2,  qw_diff2 ); 
                    sum_sq_dist += ( qw_sq1[0] + qw_sq1[1] + qw_sq2[0] + qw_sq2[1] );
                }
                return sqrt( sum_sq_dist / this->m_dim );
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=8 ) {

                    const float64x2_t qw_x1_1  = vld1q_f64( &(this->m_x1[ i     ] ) );
                    const float64x2_t qw_x1_2  = vld1q_f64( &(this->m_x1[ i + 2 ] ) );
                    const float64x2_t qw_x1_3  = vld1q_f64( &(this->m_x1[ i + 4 ] ) );
                    const float64x2_t qw_x1_4  = vld1q_f64( &(this->m_x1[ i + 6 ] ) );
                    const float64x2_t qw_x2_1  = vld1q_f64( &(this->m_x2[ i     ] ) );
                    const float64x2_t qw_x2_2  = vld1q_f64( &(this->m_x2[ i + 2 ] ) );
                    const float64x2_t qw_x2_3  = vld1q_f64( &(this->m_x2[ i + 4 ] ) );
                    const float64x2_t qw_x2_4  = vld1q_f64( &(this->m_x2[ i + 6 ] ) );
                    const float64x2_t qw_diff1 = vsubq_f64( qw_x1_1, qw_x2_1 );
                    const float64x2_t qw_diff2 = vsubq_f64( qw_x1_2, qw_x2_2 );
                    const float64x2_t qw_diff3 = vsubq_f64( qw_x1_3, qw_x2_3 );
                    const float64x2_t qw_diff4 = vsubq_f64( qw_x1_4, qw_x2_4 );
                    const float64x2_t qw_sq1   = vmulq_f64( qw_diff1,  qw_diff1 ); 
                    const float64x2_t qw_sq2   = vmulq_f64( qw_diff2,  qw_diff2 ); 
                    const float64x2_t qw_sq3   = vmulq_f64( qw_diff3,  qw_diff3 ); 
                    const float64x2_t qw_sq4   = vmulq_f64( qw_diff4,  qw_diff4 ); 
                    sum_sq_dist += (    qw_sq1[0] + qw_sq1[1] + qw_sq2[0] + qw_sq2[1]
                                     +  qw_sq3[0] + qw_sq3[1] + qw_sq4[0] + qw_sq4[1] );
                }
                return sqrt( sum_sq_dist / this->m_dim );
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=16 ) {

                    const float64x2_t qw_x1_1  = vld1q_f64( &(this->m_x1[ i      ] ) );
                    const float64x2_t qw_x1_2  = vld1q_f64( &(this->m_x1[ i +  2 ] ) );
                    const float64x2_t qw_x1_3  = vld1q_f64( &(this->m_x1[ i +  4 ] ) );
                    const float64x2_t qw_x1_4  = vld1q_f64( &(this->m_x1[ i +  6 ] ) );
                    const float64x2_t qw_x1_5  = vld1q_f64( &(this->m_x1[ i +  8 ] ) );
                    const float64x2_t qw_x1_6  = vld1q_f64( &(this->m_x1[ i + 10 ] ) );
                    const float64x2_t qw_x1_7  = vld1q_f64( &(this->m_x1[ i + 12 ] ) );
                    const float64x2_t qw_x1_8  = vld1q_f64( &(this->m_x1[ i + 14 ] ) );
                    const float64x2_t qw_x2_1  = vld1q_f64( &(this->m_x2[ i      ] ) );
                    const float64x2_t qw_x2_2  = vld1q_f64( &(this->m_x2[ i +  2 ] ) );
                    const float64x2_t qw_x2_3  = vld1q_f64( &(this->m_x2[ i +  4 ] ) );
                    const float64x2_t qw_x2_4  = vld1q_f64( &(this->m_x2[ i +  6 ] ) );
                    const float64x2_t qw_x2_5  = vld1q_f64( &(this->m_x2[ i +  8 ] ) );
                    const float64x2_t qw_x2_6  = vld1q_f64( &(this->m_x2[ i + 10 ] ) );
                    const float64x2_t qw_x2_7  = vld1q_f64( &(this->m_x2[ i + 12 ] ) );
                    const float64x2_t qw_x2_8  = vld1q_f64( &(this->m_x2[ i + 14 ] ) );
                    const float64x2_t qw_diff1 = vsubq_f64( qw_x1_1, qw_x2_1 );
                    const float64x2_t qw_diff2 = vsubq_f64( qw_x1_2, qw_x2_2 );
                    const float64x2_t qw_diff3 = vsubq_f64( qw_x1_3, qw_x2_3 );
                    const float64x2_t qw_diff4 = vsubq_f64( qw_x1_4, qw_x2_4 );
                    const float64x2_t qw_diff5 = vsubq_f64( qw_x1_5, qw_x2_5 );
                    const float64x2_t qw_diff6 = vsubq_f64( qw_x1_6, qw_x2_6 );
                    const float64x2_t qw_diff7 = vsubq_f64( qw_x1_7, qw_x2_7 );
                    const float64x2_t qw_diff8 = vsubq_f64( qw_x1_8, qw_x2_8 );
                    const float64x2_t qw_sq1   = vmulq_f64( qw_diff1,  qw_diff1 ); 
                    const float64x2_t qw_sq2   = vmulq_f64( qw_diff2,  qw_diff2 ); 
                    const float64x2_t qw_sq3   = vmulq_f64( qw_diff3,  qw_diff3 ); 
                    const float64x2_t qw_sq4   = vmulq_f64( qw_diff4,  qw_diff4 ); 
                    const float64x2_t qw_sq5   = vmulq_f64( qw_diff5,  qw_diff5 ); 
                    const float64x2_t qw_sq6   = vmulq_f64( qw_diff6,  qw_diff6 ); 
                    const float64x2_t qw_sq7   = vmulq_f64( qw_diff7,  qw_diff7 ); 
                    const float64x2_t qw_sq8   = vmulq_f64( qw_diff8,  qw_diff8 ); 
                    sum_sq_dist += (    qw_sq1[0] + qw_sq1[1] + qw_sq2[0] + qw_sq2[1]
                                     +  qw_sq3[0] + qw_sq3[1] + qw_sq4[0] + qw_sq4[1]
                                     +  qw_sq5[0] + qw_sq5[1] + qw_sq6[0] + qw_sq6[1]
                                     +  qw_sq7[0] + qw_sq7[1] + qw_sq8[0] + qw_sq8[1] );
                }
                return sqrt( sum_sq_dist / this->m_dim );
            }
        }
        return 0.0; // not reachable
    }

    virtual T getDistX1X2Neon()
    {
        if constexpr ( is_same< float,T >::value ) {

            if ( m_factor_loop_unrolling == 1 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=4 ) {

                    const float32x4_t qw_x1_1  = vld1q_f32( &(this->m_x1[ i     ] ) );
                    const float32x4_t qw_x2_1  = vld1q_f32( &(this->m_x2[ i     ] ) );
                    const float32x4_t qw_diff1 = vsubq_f32( qw_x1_1, qw_x2_1 );
                    const float32x4_t qw_sq1   = vmulq_f32( qw_diff1,  qw_diff1 ); 
                    sum_sq_dist += ( qw_sq1[0] + qw_sq1[1] + qw_sq1[2] + qw_sq1[3] );
                }
                return sqrt( sum_sq_dist );
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=8 ) {

                    const float32x4_t qw_x1_1  = vld1q_f32( &(this->m_x1[ i     ] ) );
                    const float32x4_t qw_x1_2  = vld1q_f32( &(this->m_x1[ i + 4 ] ) );
                    const float32x4_t qw_x2_1  = vld1q_f32( &(this->m_x2[ i     ] ) );
                    const float32x4_t qw_x2_2  = vld1q_f32( &(this->m_x2[ i + 4 ] ) );
                    const float32x4_t qw_diff1 = vsubq_f32( qw_x1_1, qw_x2_1 );
                    const float32x4_t qw_diff2 = vsubq_f32( qw_x1_2, qw_x2_2 );
                    const float32x4_t qw_sq1   = vmulq_f32( qw_diff1,  qw_diff1 ); 
                    const float32x4_t qw_sq2   = vmulq_f32( qw_diff2,  qw_diff2 ); 
                    sum_sq_dist += ( qw_sq1[0] + qw_sq1[1] + qw_sq1[2] + qw_sq1[3] + qw_sq2[0] + qw_sq2[1] + qw_sq2[2] + qw_sq2[3] );
                }
                return sqrt( sum_sq_dist );
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=16 ) {

                    const float32x4_t qw_x1_1  = vld1q_f32( &(this->m_x1[ i      ] ) );
                    const float32x4_t qw_x1_2  = vld1q_f32( &(this->m_x1[ i +  4 ] ) );
                    const float32x4_t qw_x1_3  = vld1q_f32( &(this->m_x1[ i +  8 ] ) );
                    const float32x4_t qw_x1_4  = vld1q_f32( &(this->m_x1[ i + 12 ] ) );
                    const float32x4_t qw_x2_1  = vld1q_f32( &(this->m_x2[ i      ] ) );
                    const float32x4_t qw_x2_2  = vld1q_f32( &(this->m_x2[ i +  4 ] ) );
                    const float32x4_t qw_x2_3  = vld1q_f32( &(this->m_x2[ i +  8 ] ) );
                    const float32x4_t qw_x2_4  = vld1q_f32( &(this->m_x2[ i + 12 ] ) );
                    const float32x4_t qw_diff1 = vsubq_f32( qw_x1_1, qw_x2_1 );
                    const float32x4_t qw_diff2 = vsubq_f32( qw_x1_2, qw_x2_2 );
                    const float32x4_t qw_diff3 = vsubq_f32( qw_x1_3, qw_x2_3 );
                    const float32x4_t qw_diff4 = vsubq_f32( qw_x1_4, qw_x2_4 );
                    const float32x4_t qw_sq1   = vmulq_f32( qw_diff1,  qw_diff1 ); 
                    const float32x4_t qw_sq2   = vmulq_f32( qw_diff2,  qw_diff2 ); 
                    const float32x4_t qw_sq3   = vmulq_f32( qw_diff3,  qw_diff3 ); 
                    const float32x4_t qw_sq4   = vmulq_f32( qw_diff4,  qw_diff4 ); 
                    sum_sq_dist += (   qw_sq1[0] + qw_sq1[1] + qw_sq1[2] + qw_sq1[3] + qw_sq2[0] + qw_sq2[1] + qw_sq2[2] + qw_sq2[3] 
                                     + qw_sq3[0] + qw_sq3[1] + qw_sq3[2] + qw_sq3[3] + qw_sq4[0] + qw_sq4[1] + qw_sq4[2] + qw_sq4[3] );
                }
                return sqrt( sum_sq_dist );
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=16 ) {

                    const float32x4_t qw_x1_1  = vld1q_f32( &(this->m_x1[ i      ] ) );
                    const float32x4_t qw_x1_2  = vld1q_f32( &(this->m_x1[ i +  4 ] ) );
                    const float32x4_t qw_x1_3  = vld1q_f32( &(this->m_x1[ i +  8 ] ) );
                    const float32x4_t qw_x1_4  = vld1q_f32( &(this->m_x1[ i + 12 ] ) );
                    const float32x4_t qw_x1_5  = vld1q_f32( &(this->m_x1[ i + 16 ] ) );
                    const float32x4_t qw_x1_6  = vld1q_f32( &(this->m_x1[ i + 20 ] ) );
                    const float32x4_t qw_x1_7  = vld1q_f32( &(this->m_x1[ i + 24 ] ) );
                    const float32x4_t qw_x1_8  = vld1q_f32( &(this->m_x1[ i + 28 ] ) );
                    const float32x4_t qw_x2_1  = vld1q_f32( &(this->m_x2[ i      ] ) );
                    const float32x4_t qw_x2_2  = vld1q_f32( &(this->m_x2[ i +  4 ] ) );
                    const float32x4_t qw_x2_3  = vld1q_f32( &(this->m_x2[ i +  8 ] ) );
                    const float32x4_t qw_x2_4  = vld1q_f32( &(this->m_x2[ i + 12 ] ) );
                    const float32x4_t qw_x2_5  = vld1q_f32( &(this->m_x2[ i + 16 ] ) );
                    const float32x4_t qw_x2_6  = vld1q_f32( &(this->m_x2[ i + 20 ] ) );
                    const float32x4_t qw_x2_7  = vld1q_f32( &(this->m_x2[ i + 24 ] ) );
                    const float32x4_t qw_x2_8  = vld1q_f32( &(this->m_x2[ i + 28 ] ) );
                    const float32x4_t qw_diff1 = vsubq_f32( qw_x1_1, qw_x2_1 );
                    const float32x4_t qw_diff2 = vsubq_f32( qw_x1_2, qw_x2_2 );
                    const float32x4_t qw_diff3 = vsubq_f32( qw_x1_3, qw_x2_3 );
                    const float32x4_t qw_diff4 = vsubq_f32( qw_x1_4, qw_x2_4 );
                    const float32x4_t qw_diff5 = vsubq_f32( qw_x1_5, qw_x2_5 );
                    const float32x4_t qw_diff6 = vsubq_f32( qw_x1_6, qw_x2_6 );
                    const float32x4_t qw_diff7 = vsubq_f32( qw_x1_7, qw_x2_7 );
                    const float32x4_t qw_diff8 = vsubq_f32( qw_x1_8, qw_x2_8 );
                    const float32x4_t qw_sq1   = vmulq_f32( qw_diff1,  qw_diff1 ); 
                    const float32x4_t qw_sq2   = vmulq_f32( qw_diff2,  qw_diff2 ); 
                    const float32x4_t qw_sq3   = vmulq_f32( qw_diff3,  qw_diff3 ); 
                    const float32x4_t qw_sq4   = vmulq_f32( qw_diff4,  qw_diff4 ); 
                    const float32x4_t qw_sq5   = vmulq_f32( qw_diff5,  qw_diff5 ); 
                    const float32x4_t qw_sq6   = vmulq_f32( qw_diff6,  qw_diff6 ); 
                    const float32x4_t qw_sq7   = vmulq_f32( qw_diff7,  qw_diff7 ); 
                    const float32x4_t qw_sq8   = vmulq_f32( qw_diff8,  qw_diff8 ); 
                    sum_sq_dist += (   qw_sq1[0] + qw_sq1[1] + qw_sq1[2] + qw_sq1[3] + qw_sq2[0] + qw_sq2[1] + qw_sq2[2] + qw_sq2[3] 
                                     + qw_sq3[0] + qw_sq3[1] + qw_sq3[2] + qw_sq3[3] + qw_sq4[0] + qw_sq4[1] + qw_sq4[2] + qw_sq4[3]
                                     + qw_sq5[0] + qw_sq5[1] + qw_sq5[2] + qw_sq5[3] + qw_sq6[0] + qw_sq6[1] + qw_sq6[2] + qw_sq6[3] 
                                     + qw_sq7[0] + qw_sq7[1] + qw_sq7[2] + qw_sq7[3] + qw_sq8[0] + qw_sq8[1] + qw_sq8[2] + qw_sq8[3] );
                }
                return sqrt( sum_sq_dist );
            }
        }
        else {
            if ( m_factor_loop_unrolling == 1 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=4 ) {

                    const float64x2_t qw_x1_1  = vld1q_f64( &(this->m_x1[ i     ] ) );
                    const float64x2_t qw_x2_1  = vld1q_f64( &(this->m_x2[ i     ] ) );
                    const float64x2_t qw_diff1 = vsubq_f64( qw_x1_1, qw_x2_1 );
                    const float64x2_t qw_sq1   = vmulq_f64( qw_diff1,  qw_diff1 ); 
                    sum_sq_dist += ( qw_sq1[0] + qw_sq1[1] );
                }
                return sqrt( sum_sq_dist );
            }
            else if ( m_factor_loop_unrolling == 2 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=4 ) {

                    const float64x2_t qw_x1_1  = vld1q_f64( &(this->m_x1[ i     ] ) );
                    const float64x2_t qw_x1_2  = vld1q_f64( &(this->m_x1[ i + 2 ] ) );
                    const float64x2_t qw_x2_1  = vld1q_f64( &(this->m_x2[ i     ] ) );
                    const float64x2_t qw_x2_2  = vld1q_f64( &(this->m_x2[ i + 2 ] ) );
                    const float64x2_t qw_diff1 = vsubq_f64( qw_x1_1, qw_x2_1 );
                    const float64x2_t qw_diff2 = vsubq_f64( qw_x1_2, qw_x2_2 );
                    const float64x2_t qw_sq1   = vmulq_f64( qw_diff1,  qw_diff1 ); 
                    const float64x2_t qw_sq2   = vmulq_f64( qw_diff2,  qw_diff2 ); 
                    sum_sq_dist += ( qw_sq1[0] + qw_sq1[1] + qw_sq2[0] + qw_sq2[1] );
                }
                return sqrt( sum_sq_dist );
            }
            else if ( m_factor_loop_unrolling == 4 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=8 ) {

                    const float64x2_t qw_x1_1  = vld1q_f64( &(this->m_x1[ i     ] ) );
                    const float64x2_t qw_x1_2  = vld1q_f64( &(this->m_x1[ i + 2 ] ) );
                    const float64x2_t qw_x1_3  = vld1q_f64( &(this->m_x1[ i + 4 ] ) );
                    const float64x2_t qw_x1_4  = vld1q_f64( &(this->m_x1[ i + 6 ] ) );
                    const float64x2_t qw_x2_1  = vld1q_f64( &(this->m_x2[ i     ] ) );
                    const float64x2_t qw_x2_2  = vld1q_f64( &(this->m_x2[ i + 2 ] ) );
                    const float64x2_t qw_x2_3  = vld1q_f64( &(this->m_x2[ i + 4 ] ) );
                    const float64x2_t qw_x2_4  = vld1q_f64( &(this->m_x2[ i + 6 ] ) );
                    const float64x2_t qw_diff1 = vsubq_f64( qw_x1_1, qw_x2_1 );
                    const float64x2_t qw_diff2 = vsubq_f64( qw_x1_2, qw_x2_2 );
                    const float64x2_t qw_diff3 = vsubq_f64( qw_x1_3, qw_x2_3 );
                    const float64x2_t qw_diff4 = vsubq_f64( qw_x1_4, qw_x2_4 );
                    const float64x2_t qw_sq1   = vmulq_f64( qw_diff1,  qw_diff1 ); 
                    const float64x2_t qw_sq2   = vmulq_f64( qw_diff2,  qw_diff2 ); 
                    const float64x2_t qw_sq3   = vmulq_f64( qw_diff3,  qw_diff3 ); 
                    const float64x2_t qw_sq4   = vmulq_f64( qw_diff4,  qw_diff4 ); 
                    sum_sq_dist += (    qw_sq1[0] + qw_sq1[1] + qw_sq2[0] + qw_sq2[1]
                                     +  qw_sq3[0] + qw_sq3[1] + qw_sq4[0] + qw_sq4[1] );
                }
                return sqrt( sum_sq_dist );
            }
            else if ( m_factor_loop_unrolling == 8 ) {

                T sum_sq_dist = 0.0;

                for ( int i = 0; i < this->m_dim; i+=16 ) {

                    const float64x2_t qw_x1_1  = vld1q_f64( &(this->m_x1[ i      ] ) );
                    const float64x2_t qw_x1_2  = vld1q_f64( &(this->m_x1[ i +  2 ] ) );
                    const float64x2_t qw_x1_3  = vld1q_f64( &(this->m_x1[ i +  4 ] ) );
                    const float64x2_t qw_x1_4  = vld1q_f64( &(this->m_x1[ i +  6 ] ) );
                    const float64x2_t qw_x1_5  = vld1q_f64( &(this->m_x1[ i +  8 ] ) );
                    const float64x2_t qw_x1_6  = vld1q_f64( &(this->m_x1[ i + 10 ] ) );
                    const float64x2_t qw_x1_7  = vld1q_f64( &(this->m_x1[ i + 12 ] ) );
                    const float64x2_t qw_x1_8  = vld1q_f64( &(this->m_x1[ i + 14 ] ) );
                    const float64x2_t qw_x2_1  = vld1q_f64( &(this->m_x2[ i      ] ) );
                    const float64x2_t qw_x2_2  = vld1q_f64( &(this->m_x2[ i +  2 ] ) );
                    const float64x2_t qw_x2_3  = vld1q_f64( &(this->m_x2[ i +  4 ] ) );
                    const float64x2_t qw_x2_4  = vld1q_f64( &(this->m_x2[ i +  6 ] ) );
                    const float64x2_t qw_x2_5  = vld1q_f64( &(this->m_x2[ i +  8 ] ) );
                    const float64x2_t qw_x2_6  = vld1q_f64( &(this->m_x2[ i + 10 ] ) );
                    const float64x2_t qw_x2_7  = vld1q_f64( &(this->m_x2[ i + 12 ] ) );
                    const float64x2_t qw_x2_8  = vld1q_f64( &(this->m_x2[ i + 14 ] ) );
                    const float64x2_t qw_diff1 = vsubq_f64( qw_x1_1, qw_x2_1 );
                    const float64x2_t qw_diff2 = vsubq_f64( qw_x1_2, qw_x2_2 );
                    const float64x2_t qw_diff3 = vsubq_f64( qw_x1_3, qw_x2_3 );
                    const float64x2_t qw_diff4 = vsubq_f64( qw_x1_4, qw_x2_4 );
                    const float64x2_t qw_diff5 = vsubq_f64( qw_x1_5, qw_x2_5 );
                    const float64x2_t qw_diff6 = vsubq_f64( qw_x1_6, qw_x2_6 );
                    const float64x2_t qw_diff7 = vsubq_f64( qw_x1_7, qw_x2_7 );
                    const float64x2_t qw_diff8 = vsubq_f64( qw_x1_8, qw_x2_8 );
                    const float64x2_t qw_sq1   = vmulq_f64( qw_diff1,  qw_diff1 ); 
                    const float64x2_t qw_sq2   = vmulq_f64( qw_diff2,  qw_diff2 ); 
                    const float64x2_t qw_sq3   = vmulq_f64( qw_diff3,  qw_diff3 ); 
                    const float64x2_t qw_sq4   = vmulq_f64( qw_diff4,  qw_diff4 ); 
                    const float64x2_t qw_sq5   = vmulq_f64( qw_diff5,  qw_diff5 ); 
                    const float64x2_t qw_sq6   = vmulq_f64( qw_diff6,  qw_diff6 ); 
                    const float64x2_t qw_sq7   = vmulq_f64( qw_diff7,  qw_diff7 ); 
                    const float64x2_t qw_sq8   = vmulq_f64( qw_diff8,  qw_diff8 ); 
                    sum_sq_dist += (    qw_sq1[0] + qw_sq1[1] + qw_sq2[0] + qw_sq2[1]
                                     +  qw_sq3[0] + qw_sq3[1] + qw_sq4[0] + qw_sq4[1]
                                     +  qw_sq5[0] + qw_sq5[1] + qw_sq6[0] + qw_sq6[1]
                                     +  qw_sq7[0] + qw_sq7[1] + qw_sq8[0] + qw_sq8[1] );
                }
                return sqrt( sum_sq_dist );
            }
        }
        return 0.0; // not reachable
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseJacobiSolver_multithread : public TestCaseJacobiSolver_NEON<T, IS_COL_MAJOR> {

    size_t                      m_num_threads;
    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    vector<thread>              m_threads;

  public:

    TestCaseJacobiSolver_multithread( const int dim, const int solver_iteration, const int factor_loop_unrolling, const int num_threads )
        :TestCaseJacobiSolver_NEON<T, IS_COL_MAJOR>( dim, solver_iteration, factor_loop_unrolling )
        ,m_num_threads( num_threads )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
    {
	assert ( this->m_dim % 4 == 0 && this->m_dim >= 4 );

        this->setNEON( num_threads, factor_loop_unrolling );

        const size_t num_rows_per_thread = this->m_dim / m_num_threads;

        auto thread_lambda = [ this, num_rows_per_thread ]( const size_t thread_index ) {

            const size_t row_begin = thread_index * num_rows_per_thread;
            const size_t row_end   = row_begin + num_rows_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                if constexpr ( IS_COL_MAJOR ) {
                    if ( this->m_updating_x1 ) {
                        this->calcX1ColMajor( row_begin, row_end );
                    }
                    else {
                        this->calcX2ColMajor( row_begin, row_end );
                    }
                }
                else {
                    if ( this->m_updating_x1 ) {
                        this->calcX1RowMajor( row_begin, row_end );
                    }
                    else {
                        this->calcX2RowMajor( row_begin, row_end );
                    }
                }

                m_fan_in.notify();
                if( m_fan_in.isTerminating() ) {
                    break;
                }
            }
        };

        for ( size_t i = 0; i < m_num_threads; i++ ) {

            m_threads.emplace_back( thread_lambda, i );
        }
    }

    virtual ~TestCaseJacobiSolver_multithread(){

        m_fan_out.terminate();
        m_fan_in. terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    void run() 
    {
        this->m_diff_x1_x2.clear();

        if constexpr ( is_same< float,T >::value ) {

            if ( this->m_factor_loop_unrolling == 1 ) {

                for (int i = 0; i < this->m_dim; i += 4 ) {

                    const float32x4_t qw_d1       = { this->m_D[i  ], this->m_D[i+1], this->m_D[i+2], this->m_D[i+3] };
                    const float32x4_t qw_d_inv1_1 = vrecpeq_f32( qw_d1 );
                    const float32x4_t qw_d_inv2_1 = vmulq_f32( vrecpsq_f32( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );

                    memcpy(&(this->m_Dinv[i  ]), &qw_d_inv2_1, sizeof(float)*4);
                }
            }
            else if ( this->m_factor_loop_unrolling == 2 ) {

                for ( int i = 0; i < this->m_dim; i+=8 ) {

                    const float32x4_t qw_d1       = { this->m_D[i  ], this->m_D[i+1], this->m_D[i+2], this->m_D[i+3] };
                    const float32x4_t qw_d2       = { this->m_D[i+4], this->m_D[i+5], this->m_D[i+6], this->m_D[i+7] };
                    const float32x4_t qw_d_inv1_1 = vrecpeq_f32( qw_d1 );
                    const float32x4_t qw_d_inv1_2 = vrecpeq_f32( qw_d2 );
                    const float32x4_t qw_d_inv2_1 = vmulq_f32( vrecpsq_f32( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float32x4_t qw_d_inv2_2 = vmulq_f32( vrecpsq_f32( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );

                    memcpy(&(this->m_Dinv[i  ]), &qw_d_inv2_1, sizeof(float)*4);
                    memcpy(&(this->m_Dinv[i+4]), &qw_d_inv2_2, sizeof(float)*4);
                }
            }
            else if ( this->m_factor_loop_unrolling == 4 ) {

                for ( int i = 0; i < this->m_dim; i+=16 ) {

                    const float32x4_t qw_d1       = { this->m_D[i   ], this->m_D[i+ 1], this->m_D[i+ 2], this->m_D[i+ 3] };
                    const float32x4_t qw_d2       = { this->m_D[i+ 4], this->m_D[i+ 5], this->m_D[i+ 6], this->m_D[i+ 7] };
                    const float32x4_t qw_d3       = { this->m_D[i+ 8], this->m_D[i+ 9], this->m_D[i+10], this->m_D[i+11] };
                    const float32x4_t qw_d4       = { this->m_D[i+12], this->m_D[i+13], this->m_D[i+14], this->m_D[i+15] };
                    const float32x4_t qw_d_inv1_1 = vrecpeq_f32( qw_d1 );
                    const float32x4_t qw_d_inv1_2 = vrecpeq_f32( qw_d2 );
                    const float32x4_t qw_d_inv1_3 = vrecpeq_f32( qw_d3 );
                    const float32x4_t qw_d_inv1_4 = vrecpeq_f32( qw_d4 );
                    const float32x4_t qw_d_inv2_1 = vmulq_f32( vrecpsq_f32( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float32x4_t qw_d_inv2_2 = vmulq_f32( vrecpsq_f32( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float32x4_t qw_d_inv2_3 = vmulq_f32( vrecpsq_f32( qw_d3, qw_d_inv1_3 ), qw_d_inv1_3 );
                    const float32x4_t qw_d_inv2_4 = vmulq_f32( vrecpsq_f32( qw_d4, qw_d_inv1_4 ), qw_d_inv1_4 );

                    memcpy(&(this->m_Dinv[i   ]), &qw_d_inv2_1, sizeof(float)*4);
                    memcpy(&(this->m_Dinv[i+ 4]), &qw_d_inv2_2, sizeof(float)*4);
                    memcpy(&(this->m_Dinv[i+ 8]), &qw_d_inv2_3, sizeof(float)*4);
                    memcpy(&(this->m_Dinv[i+12]), &qw_d_inv2_4, sizeof(float)*4);
                }
            }
        }
        else {
            if ( this->m_factor_loop_unrolling == 1 ) {

                for ( int i = 0; i < this->m_dim; i+=2 ) {

                    const float64x2_t qw_d1      = { this->m_D[i  ], this->m_D[i+1] };
                    const float64x2_t qw_d_inv1_1 = vrecpeq_f64( qw_d1 );
                    const float64x2_t qw_d_inv2_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float64x2_t qw_d_inv3_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv2_1 ), qw_d_inv2_1 );

                    memcpy(&(this->m_Dinv[i  ]), &qw_d_inv3_1, sizeof(double)*2);
                }
            }
            if ( this->m_factor_loop_unrolling == 2 ) {

                for ( int i = 0; i < this->m_dim; i+=4 ) {

                    const float64x2_t qw_d1      = { this->m_D[i  ], this->m_D[i+1] };
                    const float64x2_t qw_d2      = { this->m_D[i+2], this->m_D[i+3] };
                    const float64x2_t qw_d_inv1_1 = vrecpeq_f64( qw_d1 );
                    const float64x2_t qw_d_inv1_2 = vrecpeq_f64( qw_d2 );
                    const float64x2_t qw_d_inv2_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float64x2_t qw_d_inv2_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float64x2_t qw_d_inv3_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv2_1 ), qw_d_inv2_1 );
                    const float64x2_t qw_d_inv3_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv2_2 ), qw_d_inv2_2 );

                    memcpy(&(this->m_Dinv[i  ]), &qw_d_inv3_1, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+2]), &qw_d_inv3_2, sizeof(double)*2);
                }
            }
            else if ( this->m_factor_loop_unrolling == 4 ) {

                for ( int i = 0; i < this->m_dim; i+=8 ) {

                    const float64x2_t qw_d1      = { this->m_D[i  ], this->m_D[i+1] };
                    const float64x2_t qw_d2      = { this->m_D[i+2], this->m_D[i+3] };
                    const float64x2_t qw_d3      = { this->m_D[i+4], this->m_D[i+5] };
                    const float64x2_t qw_d4      = { this->m_D[i+6], this->m_D[i+7] };
                    const float64x2_t qw_d_inv1_1 = vrecpeq_f64( qw_d1 );
                    const float64x2_t qw_d_inv1_2 = vrecpeq_f64( qw_d2 );
                    const float64x2_t qw_d_inv1_3 = vrecpeq_f64( qw_d3 );
                    const float64x2_t qw_d_inv1_4 = vrecpeq_f64( qw_d4 );
                    const float64x2_t qw_d_inv2_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float64x2_t qw_d_inv2_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float64x2_t qw_d_inv2_3 = vmulq_f64( vrecpsq_f64( qw_d3, qw_d_inv1_3 ), qw_d_inv1_3 );
                    const float64x2_t qw_d_inv2_4 = vmulq_f64( vrecpsq_f64( qw_d4, qw_d_inv1_4 ), qw_d_inv1_4 );
                    const float64x2_t qw_d_inv3_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv2_1 ), qw_d_inv2_1 );
                    const float64x2_t qw_d_inv3_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv2_2 ), qw_d_inv2_2 );
                    const float64x2_t qw_d_inv3_3 = vmulq_f64( vrecpsq_f64( qw_d3, qw_d_inv2_3 ), qw_d_inv2_3 );
                    const float64x2_t qw_d_inv3_4 = vmulq_f64( vrecpsq_f64( qw_d4, qw_d_inv2_4 ), qw_d_inv2_4 );

                    memcpy(&(this->m_Dinv[i  ]), &qw_d_inv3_1, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+2]), &qw_d_inv3_2, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+4]), &qw_d_inv3_3, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+6]), &qw_d_inv3_4, sizeof(double)*2);
                }
            }
            else if ( this->m_factor_loop_unrolling == 8 ) {

                for ( int i = 0; i < this->m_dim; i+=16 ) {

                    const float64x2_t qw_d1      = { this->m_D[i   ], this->m_D[i+ 1] };
                    const float64x2_t qw_d2      = { this->m_D[i+ 2], this->m_D[i+ 3] };
                    const float64x2_t qw_d3      = { this->m_D[i+ 4], this->m_D[i+ 5] };
                    const float64x2_t qw_d4      = { this->m_D[i+ 6], this->m_D[i+ 7] };
                    const float64x2_t qw_d5      = { this->m_D[i+ 8], this->m_D[i+ 9] };
                    const float64x2_t qw_d6      = { this->m_D[i+10], this->m_D[i+11] };
                    const float64x2_t qw_d7      = { this->m_D[i+12], this->m_D[i+13] };
                    const float64x2_t qw_d8      = { this->m_D[i+14], this->m_D[i+15] };
                    const float64x2_t qw_d_inv1_1 = vrecpeq_f64( qw_d1 );
                    const float64x2_t qw_d_inv1_2 = vrecpeq_f64( qw_d2 );
                    const float64x2_t qw_d_inv1_3 = vrecpeq_f64( qw_d3 );
                    const float64x2_t qw_d_inv1_4 = vrecpeq_f64( qw_d4 );
                    const float64x2_t qw_d_inv1_5 = vrecpeq_f64( qw_d5 );
                    const float64x2_t qw_d_inv1_6 = vrecpeq_f64( qw_d6 );
                    const float64x2_t qw_d_inv1_7 = vrecpeq_f64( qw_d7 );
                    const float64x2_t qw_d_inv1_8 = vrecpeq_f64( qw_d8 );
                    const float64x2_t qw_d_inv2_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv1_1 ), qw_d_inv1_1 );
                    const float64x2_t qw_d_inv2_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv1_2 ), qw_d_inv1_2 );
                    const float64x2_t qw_d_inv2_3 = vmulq_f64( vrecpsq_f64( qw_d3, qw_d_inv1_3 ), qw_d_inv1_3 );
                    const float64x2_t qw_d_inv2_4 = vmulq_f64( vrecpsq_f64( qw_d4, qw_d_inv1_4 ), qw_d_inv1_4 );
                    const float64x2_t qw_d_inv2_5 = vmulq_f64( vrecpsq_f64( qw_d5, qw_d_inv1_5 ), qw_d_inv1_5 );
                    const float64x2_t qw_d_inv2_6 = vmulq_f64( vrecpsq_f64( qw_d6, qw_d_inv1_6 ), qw_d_inv1_6 );
                    const float64x2_t qw_d_inv2_7 = vmulq_f64( vrecpsq_f64( qw_d7, qw_d_inv1_7 ), qw_d_inv1_7 );
                    const float64x2_t qw_d_inv2_8 = vmulq_f64( vrecpsq_f64( qw_d8, qw_d_inv1_8 ), qw_d_inv1_8 );
                    const float64x2_t qw_d_inv3_1 = vmulq_f64( vrecpsq_f64( qw_d1, qw_d_inv2_1 ), qw_d_inv2_1 );
                    const float64x2_t qw_d_inv3_2 = vmulq_f64( vrecpsq_f64( qw_d2, qw_d_inv2_2 ), qw_d_inv2_2 );
                    const float64x2_t qw_d_inv3_3 = vmulq_f64( vrecpsq_f64( qw_d3, qw_d_inv2_3 ), qw_d_inv2_3 );
                    const float64x2_t qw_d_inv3_4 = vmulq_f64( vrecpsq_f64( qw_d4, qw_d_inv2_4 ), qw_d_inv2_4 );
                    const float64x2_t qw_d_inv3_5 = vmulq_f64( vrecpsq_f64( qw_d5, qw_d_inv2_5 ), qw_d_inv2_5 );
                    const float64x2_t qw_d_inv3_6 = vmulq_f64( vrecpsq_f64( qw_d6, qw_d_inv2_6 ), qw_d_inv2_6 );
                    const float64x2_t qw_d_inv3_7 = vmulq_f64( vrecpsq_f64( qw_d7, qw_d_inv2_7 ), qw_d_inv2_7 );
                    const float64x2_t qw_d_inv3_8 = vmulq_f64( vrecpsq_f64( qw_d8, qw_d_inv2_8 ), qw_d_inv2_8 );

                    memcpy(&(this->m_Dinv[i   ]), &qw_d_inv3_1, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+ 2]), &qw_d_inv3_2, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+ 4]), &qw_d_inv3_3, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+ 6]), &qw_d_inv3_4, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+ 8]), &qw_d_inv3_5, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+10]), &qw_d_inv3_6, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+12]), &qw_d_inv3_7, sizeof(double)*2);
                    memcpy(&(this->m_Dinv[i+14]), &qw_d_inv3_8, sizeof(double)*2);
                }
            }
        }
       
        for (int i = 0; i < this->m_iteration; i++ ) {

            m_fan_out.notify();
            m_fan_in. wait();

            this->m_updating_x1 = ! this->m_updating_x1;

            //T err = this->getRmsX1X2Neon();
            T err = this->getDistX1X2Neon();
            this->m_diff_x1_x2.push_back( err );
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseJacobiSolver_vDSP : public TestCaseJacobiSolver< T, IS_COL_MAJOR > {

    T* m_Dinv;
    T* m_ones;
    T* m_sums;

  public:

    TestCaseJacobiSolver_vDSP( const int dim, const int solver_iteration )
        :TestCaseJacobiSolver< T, IS_COL_MAJOR >( dim, solver_iteration )
        ,m_Dinv( new T[dim] )
        ,m_ones( new T[dim] )
        ,m_sums( new T[dim] )
    {
        static_assert( !IS_COL_MAJOR );

        this->setImplementationType( VDSP );

        for  (int i = 0 ; i < this->m_dim ; i++ ) {
            m_ones[i] = 1.0;
        }
    }

    virtual ~TestCaseJacobiSolver_vDSP() {
        delete[] m_Dinv;
        delete[] m_ones;
        delete[] m_sums;
    }

    virtual T getRmsX1X2vDSP() {
        if constexpr ( is_same< float,T >::value ) {

            vDSP_vsub( this->m_x1, 1, this->m_x2, 1, m_sums, 1, this->m_dim );
            T dot;
            vDSP_dotpr(m_sums, 1, m_sums, 1, &dot, this->m_dim );

            return sqrt( dot / this->m_dim );
        }
        else {
            vDSP_vsubD( this->m_x1, 1, this->m_x2, 1, m_sums, 1, this->m_dim );
            T dot;
            vDSP_dotprD(m_sums, 1, m_sums, 1, &dot, this->m_dim );
            return sqrt( dot / this->m_dim );

        }
    }

    virtual T getDistX1X2vDSP() {
        if constexpr ( is_same< float,T >::value ) {

            vDSP_vsub( this->m_x1, 1, this->m_x2, 1, m_sums, 1, this->m_dim );
            T dot;
            vDSP_dotpr(m_sums, 1, m_sums, 1, &dot, this->m_dim );

            return sqrt( dot );
        }
        else {
            vDSP_vsubD( this->m_x1, 1, this->m_x2, 1, m_sums, 1, this->m_dim );
            T dot;
            vDSP_dotprD(m_sums, 1, m_sums, 1, &dot, this->m_dim );
            return sqrt( dot );

        }
    }


    virtual void run() {

        this->m_diff_x1_x2.clear();

        if constexpr ( is_same< float,T >::value ) {

            vDSP_vdiv( this->m_D, 1, m_ones, 1, m_Dinv, 1, this->m_dim );

            for (int i = 0; i < this->m_iteration; i++ ) {

                if ( this->m_updating_x1 ) {
                    vDSP_mmul( this->m_A, 1, this->m_x2, 1, m_sums, 1, this->m_dim, 1, this->m_dim );
                    vDSP_vsbm( this->m_b, 1, m_sums, 1, m_Dinv, 1, this->m_x1, 1, this->m_dim );
                }
                else {
                    vDSP_mmul( this->m_A, 1, this->m_x1, 1, m_sums, 1, this->m_dim, 1, this->m_dim );
                    vDSP_vsbm( this->m_b, 1, m_sums, 1, m_Dinv, 1, this->m_x2, 1, this->m_dim );
                }

                //T err = this->getRmsX1X2vDSP();
                T err = this->getDistX1X2vDSP();

                this->m_diff_x1_x2.push_back(err);

                this->m_updating_x1 = ! this->m_updating_x1;
            }
        }
        else {

            vDSP_vdivD( this->m_D, 1, m_ones, 1, m_Dinv, 1, this->m_dim );

            for (int i = 0; i < this->m_iteration; i++ ) {

                if ( this->m_updating_x1 ) {
                    vDSP_mmulD( this->m_A, 1, this->m_x2, 1, m_sums, 1, this->m_dim, 1, this->m_dim );
                    vDSP_vsbmD( this->m_b, 1, m_sums, 1, m_Dinv, 1, this->m_x1, 1, this->m_dim );
                }
                else {
                    vDSP_mmulD( this->m_A, 1, this->m_x1, 1, m_sums, 1, this->m_dim, 1, this->m_dim );
                    vDSP_vsbmD( this->m_b, 1, m_sums, 1, m_Dinv, 1, this->m_x2, 1, this->m_dim );
                }

                //T err = this->getRmsX1X2vDSP();
                T err = this->getDistX1X2vDSP();

                this->m_diff_x1_x2.push_back(err);

                this->m_updating_x1 = ! this->m_updating_x1;
            }
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseJacobiSolver_blas : public TestCaseJacobiSolver< T, IS_COL_MAJOR > {

    T* m_Dinv;
    T* m_ones;
    T* m_sums;

  public:

    TestCaseJacobiSolver_blas( const int dim, const int solver_iteration )
        :TestCaseJacobiSolver< T, IS_COL_MAJOR >( dim, solver_iteration )
        ,m_Dinv( new T[dim] )
        ,m_ones( new T[dim] )
        ,m_sums( new T[dim] )
    {
        this->setImplementationType( BLAS );

        for  (int i = 0 ; i < this->m_dim ; i++ ) {
            m_ones[i] = 1.0;
        }
    }

    virtual ~TestCaseJacobiSolver_blas() {
        delete[] m_Dinv;
        delete[] m_ones;
        delete[] m_sums;
    }

    virtual T getRmsX1X2vDSP() {

        if constexpr ( is_same< float,T >::value ) {

            vDSP_vsub( this->m_x1, 1, this->m_x2, 1, m_sums, 1, this->m_dim );
            T dot;
            vDSP_dotpr(m_sums, 1, m_sums, 1, &dot, this->m_dim );

            return sqrt( dot / this->m_dim );
        }
        else {
            vDSP_vsubD( this->m_x1, 1, this->m_x2, 1, m_sums, 1, this->m_dim );
            T dot;
            vDSP_dotprD(m_sums, 1, m_sums, 1, &dot, this->m_dim );
            return sqrt( dot / this->m_dim );

        }
    }

    virtual T getDistX1X2vDSP() {

        if constexpr ( is_same< float,T >::value ) {

            vDSP_vsub( this->m_x1, 1, this->m_x2, 1, m_sums, 1, this->m_dim );
            T dot;
            vDSP_dotpr(m_sums, 1, m_sums, 1, &dot, this->m_dim );

            return sqrt( dot );
        }
        else {
            vDSP_vsubD( this->m_x1, 1, this->m_x2, 1, m_sums, 1, this->m_dim );
            T dot;
            vDSP_dotprD(m_sums, 1, m_sums, 1, &dot, this->m_dim );
            return sqrt( dot );

        }
    }

    virtual void run() {

        this->m_diff_x1_x2.clear();

        if constexpr ( is_same< float,T >::value ) {

            vDSP_vdiv( this->m_D, 1, m_ones, 1, m_Dinv, 1, this->m_dim );

            const enum CBLAS_ORDER blas_order = IS_COL_MAJOR ? CblasColMajor : CblasRowMajor;

            for (int i = 0; i < this->m_iteration; i++ ) {

                if ( this->m_updating_x1 ) {

                    cblas_sgemv( blas_order, CblasNoTrans, this->m_dim,  this->m_dim, 1.0, this->m_A, this->m_dim, this->m_x2, 1, 1.0, m_sums, 1 );
                    vDSP_vsbm( this->m_b, 1, m_sums, 1, m_Dinv, 1, this->m_x1, 1, this->m_dim );
                }
                else {
                    cblas_sgemv( blas_order, CblasNoTrans, this->m_dim,  this->m_dim, 1.0, this->m_A, this->m_dim, this->m_x1, 1, 1.0, m_sums, 1 );
                    vDSP_vsbm( this->m_b, 1, m_sums, 1, m_Dinv, 1, this->m_x2, 1, this->m_dim );
                }

                //T err = this->getRmsX1X2vDSP();
                T err = this->getDistX1X2vDSP();

                this->m_diff_x1_x2.push_back(err);

                this->m_updating_x1 = ! this->m_updating_x1;
            }
        }
        else {
            vDSP_vdivD( this->m_D, 1, m_ones, 1, m_Dinv, 1, this->m_dim );

            const enum CBLAS_ORDER blas_order = IS_COL_MAJOR ? CblasColMajor : CblasRowMajor;

            for (int i = 0; i < this->m_iteration; i++ ) {

                if ( this->m_updating_x1 ) {

                    cblas_dgemv( blas_order, CblasNoTrans, this->m_dim,  this->m_dim, 1.0, this->m_A, this->m_dim, this->m_x2, 1, 1.0, m_sums, 1 );
                    vDSP_vsbmD( this->m_b, 1, m_sums, 1, m_Dinv, 1, this->m_x1, 1, this->m_dim );
                }
                else {
                    cblas_dgemv( blas_order, CblasNoTrans, this->m_dim,  this->m_dim, 1.0, this->m_A, this->m_dim, this->m_x1, 1, 1.0, m_sums, 1 );
                    vDSP_vsbmD( this->m_b, 1, m_sums, 1, m_Dinv, 1, this->m_x2, 1, this->m_dim );
                }

                //T err = this->getRmsX1X2vDSP();
                T err = this->getDistX1X2vDSP();

                this->m_diff_x1_x2.push_back(err);

                this->m_updating_x1 = ! this->m_updating_x1;
            }
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseJacobiSolver_metal : public TestCaseJacobiSolver< T, IS_COL_MAJOR > {

    JacobiSolverMetalCpp m_metal;

  public:

    TestCaseJacobiSolver_metal( const int dim, const int solver_iteration )
        :TestCaseJacobiSolver< T, IS_COL_MAJOR >( dim, solver_iteration )
        ,m_metal( dim, solver_iteration, IS_COL_MAJOR ? 0 : 1 )
    {
        static_assert ( is_same< float,T >::value );

        this->setMetal( DEFAULT, 1, 1 );
    }

    virtual ~TestCaseJacobiSolver_metal() {;}

    virtual void run() {
         m_metal.performComputation();
         //cerr << "metal rms(x2-x1): " << sqrt(m_metal.getError() / ((float)(this->m_dim))) << "\n";;
    }

    virtual void setInitialStates( T* A, T* D, T* b , T* x1, T* x2 ) {
        m_metal.setInitialStates( A, D, b, x1, x2 );
        TestCaseJacobiSolver<T, IS_COL_MAJOR>::setInitialStates( A, D, b ,x1, x2 );
    }

    virtual T* getActiveX() {

        return m_metal.getRawPointerActiveX();
    }
};


template <class T, bool IS_COL_MAJOR>
class TestExecutorJacobiSolver : public TestExecutor {

  protected:
    const int             m_dim;
    default_random_engine m_e;
    T*                    m_A;
    T*                    m_D;
    T*                    m_b;
    T*                    m_x1;
    T*                    m_x2;
    T*                    m_x_baseline;


  public:

    TestExecutorJacobiSolver (
        ostream&   os,
        const int  dim,
        const T    condition_num,
        const T    val_low,
        const T    val_high,
        const int  num_trials,
        const bool repeatable
    )
        :TestExecutor   ( os, num_trials )
        ,m_dim          ( dim )
        ,m_e            ( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_A            ( new T [ dim * dim ] )
        ,m_D            ( new T [ dim ]       )
        ,m_b            ( new T [ dim ]       )
        ,m_x1           ( new T [ dim ]       )
        ,m_x2           ( new T [ dim ]       )
        ,m_x_baseline   ( new T [ dim ]       )
    {

        generateRandomPDMat<T, IS_COL_MAJOR>( m_A, m_dim, condition_num, m_e );
        fillArrayWithRandomValues( m_e, m_b, m_dim, val_low, val_high );

        // moving the diagonal elements from A to D.
        for ( int i = 0; i < dim ; i ++ ) {

            m_D[i] = m_A[ linear_index_mat<IS_COL_MAJOR>( i, i, dim, dim ) ];
            m_A[ linear_index_mat<IS_COL_MAJOR>( i, i, dim, dim) ] = 0.0;
        }
    }

    void prepareForRun ( const int test_case, const int num ) {

        memset( m_x1, 0, sizeof(T) * m_dim );
        memset( m_x2, 0, sizeof(T) * m_dim );

        auto t = dynamic_pointer_cast< TestCaseJacobiSolver<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_A, m_D, m_b, m_x1, m_x2 );
    }

    void cleanupAfterBatchRuns ( const int test_case ) {

        auto t = dynamic_pointer_cast< TestCaseJacobiSolver<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            memcpy( m_x_baseline, t->getActiveX(), sizeof(T) * m_dim );
        }

        t->compareTruth( m_x_baseline );
    }

    virtual ~TestExecutorJacobiSolver()
    {
        delete[] m_A;
        delete[] m_D;
        delete[] m_b;
        delete[] m_x1;
        delete[] m_x2;
        delete[] m_x_baseline;
    }
};


static const size_t NUM_TRIALS       = 10;
static const int    SOLVER_ITERATION = 10;

int  matrix_dims[]={ 64, 128, 256, 512, 1024, 2048, 4096, 8192 };

template<class T, bool IS_COL_MAJOR>
void testSuitePerType ( const T condition_num, const T gen_low, const T gen_high ) {

    const int neon_num_lanes = ( is_same<float, T>::value )? 4 : 2;

    for( auto& dim : matrix_dims ) {

        TestExecutorJacobiSolver<T, IS_COL_MAJOR> e( cout, dim, condition_num, gen_low, gen_high, NUM_TRIALS, false );

        e.addTestCase( make_shared< TestCaseJacobiSolver_baseline    <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION     ) );

        if ( dim >= neon_num_lanes  ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_NEON        <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION,  1  ) );
        }
        if ( dim >= 2 * neon_num_lanes  ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_NEON        <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION,  2  ) );
        }
        if ( dim >= 4 * neon_num_lanes  ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_NEON        <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION,  4  ) );
        }
        if ( dim >= 8 * neon_num_lanes  ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_NEON        <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION,  8  ) );
        }
        if ( dim >= 8 * 2 * neon_num_lanes  ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_multithread <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION,  8,  2 ) );
        }
        if ( dim >= 8 * 4 * neon_num_lanes  ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_multithread <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION,  8,  4 ) );
        }
        if ( dim >= 8 * 8 * neon_num_lanes  ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_multithread <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION,  8,  8 ) );
        }

        if constexpr ( !IS_COL_MAJOR ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_vDSP <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION ) );
        }

        e.addTestCase( make_shared< TestCaseJacobiSolver_blas <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION ) );

        if constexpr ( is_same< float,T >::value ) {
            e.addTestCase( make_shared< TestCaseJacobiSolver_metal <T, IS_COL_MAJOR> > ( dim, SOLVER_ITERATION ) );
        }

        e.execute();
    }
}


int main( int argc, char* argv[] )
{
    TestCaseWithTimeMeasurements::printHeader( cout );

    cerr << "\n\nTesting for type float in column-major.\n\n";

    testSuitePerType<float,  true  > ( 10.0, -1.0, 1.0 );

    cerr << "\n\nTesting for type float in row-major.\n\n";

    testSuitePerType<float,  false > ( 10.0, -1.0, 1.0 );

    cerr << "\n\nTesting for type double in column-major.\n\n";

    testSuitePerType<double, true  > ( 10.0, -1.0, 1.0 );

    cerr << "\n\nTesting for type double in row-major.\n\n";

    testSuitePerType<double, false > ( 10.0, -1.0, 1.0 );

    return 0;
}
