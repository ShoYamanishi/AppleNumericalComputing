#include <type_traits>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <thread>
#include <vector>
#include <assert.h>
#include <Accelerate/Accelerate.h>
#include <simd/simd.h>
#include <arm_neon.h>

#include "test_case_with_time_measurements.h"
#include "thread_synchronizer.h"

#include "nbody_metal_cpp.h"
#include "nbody_elements.h"
#include "nbody_elements_impl.h"

template< class T>
class TestCaseNBody : public TestCaseWithTimeMeasurements {

  protected:
    const T  EPSILON = 1.0e-5;
    const T  COEFF_G = 9.8;

    const size_t m_num_elements;
    const T      m_delta_t;
    const T      m_tolerance;

  public:
    TestCaseNBody( const size_t num_elements, const T delta_t,const T tolerance )
        :m_num_elements          ( num_elements )
        ,m_delta_t               ( delta_t )
        ,m_tolerance             ( tolerance )
    {
         static_type_guard_real<T>();

        if constexpr ( is_same<float, T>::value ) {

            setDataElementType( FLOAT );
        }
        else {
            assert(true);
        }
        setVerificationType( TRUE_FALSE );
    }

    virtual ~TestCaseNBody(){;}

    virtual void compareTruth( const NBodyElem<T>* const baseline ) {

        for ( size_t i = 0; i < m_num_elements; i++ ) {

            if ( ! getParticleAt(i).equalWithinTolerance( baseline[i], m_tolerance ) ) {

                this->setTrueFalse( false );
                return;
            }
        }
        this->setTrueFalse( true );
    }

    virtual void setInitialStates( const NBodyElem<T>* const aos ) = 0;
    virtual NBodyElem<T> getParticleAt( const size_t i ) = 0;
    virtual void run() = 0;
};



template<class T>
class TestCaseNBody_baselineSOA : public TestCaseNBody<T> {

  protected:

    NBodySOA<T>      m_soa;
    VelocityElem<T>* m_v_saved;

    virtual void inline bodyBodyInteraction(

        T& a0x,        T& a0y,        T& a0z,
        const T p0x,   const T p0y,   const T p0z,
        const T p1x,   const T p1y,   const T p1z,
        const T mass1,
        const T epsilon
    ) {
        const T dx = p1x - p0x;
        const T dy = p1y - p0y;
        const T dz = p1z - p0z;

        const T dist_sqr = dx*dx + dy*dy + dz*dz + epsilon; 

        T inv_dist;

        // vDSP's rsqrt. No noticeable difference in speed.
        //const int num_1 = 1;
        //vvrsqrtf( &inv_dist, &dist_sqr, &num_1 ); 

        inv_dist = 1.0 / sqrtf( dist_sqr);

        const T inv_dist_cube = inv_dist * inv_dist * inv_dist;
        const T s = mass1 * inv_dist_cube;

        a0x += (dx * s);
        a0y += (dy * s);
        a0z += (dz * s);
    }

  public:

    virtual void setInitialStates( const NBodyElem<T>* const aos )
    {
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
            m_soa.set(aos[i], i);
            m_v_saved[i] = aos[i].m_v;
        }
    }

    virtual NBodyElem<T> getParticleAt( const size_t i ){
        NBodyElem<T> e;
        m_soa.get(e, i);
        return e;
    }

    TestCaseNBody_baselineSOA( const size_t num_elements, const T delta_t, const T tolerance )
        :TestCaseNBody<T>( num_elements , delta_t, tolerance )
        ,m_soa( num_elements )
        ,m_v_saved( new VelocityElem<T>[ num_elements ] )
    {
        this->setSOA( num_elements );
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseNBody_baselineSOA()
    {
        delete[] m_v_saved;
    }

    virtual void run()
    {
        // reset the velocities.
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
            NBodyElem<T> e;            
            m_soa.get(e, i);
            e.m_v = m_v_saved[i].m_v;
            m_soa.set(e, i);
        }

        memset( m_soa.m_ax, 0, sizeof(float)*this->m_num_elements ); 
        memset( m_soa.m_ay, 0, sizeof(float)*this->m_num_elements ); 
        memset( m_soa.m_az, 0, sizeof(float)*this->m_num_elements ); 

        if ( m_soa.m_p0_is_active ) { // take out 'if' out of the for loop.

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                for ( size_t j = 0; j < this->m_num_elements ; j++ ) {

                    // NOTES:

                    // manual loop unrolling does not make it fater.
                    // Therefore, let clang take care of optimization
                    // ex. 12.6 secs @ 32K * 32K with manual loop unrolling
                    // while 9.5 secs without.
                    if ( i != j ) {
                        bodyBodyInteraction(
                            m_soa.m_ax  [i  ], m_soa.m_ay [i  ], m_soa.m_az [i  ],
                            m_soa.m_p0x [i  ], m_soa.m_p0y[i  ], m_soa.m_p0z[i  ],
                            m_soa.m_p0x [j  ], m_soa.m_p0y[j  ], m_soa.m_p0z[j  ],
                            m_soa.m_mass[j  ], this->EPSILON                      );
                    }
                }

                m_soa.m_vx[i]   += ( m_soa.m_ax[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                m_soa.m_vy[i]   += ( m_soa.m_ay[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                m_soa.m_vz[i]   += ( m_soa.m_az[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );

                m_soa.m_p1x[i] = m_soa.m_p0x[i] + m_soa.m_vx[i] * this->m_delta_t;
                m_soa.m_p1y[i] = m_soa.m_p0y[i] + m_soa.m_vy[i] * this->m_delta_t;
                m_soa.m_p1z[i] = m_soa.m_p0z[i] + m_soa.m_vz[i] * this->m_delta_t;
            }
            // m_soa.m_p0_is_active = false; commenting out for the test cases
        }
        else {
            assert(true); // this should never be called in the test cases.

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                for ( size_t j = 0; j < this->m_num_elements ; j++ ) {

                    bodyBodyInteraction(
                        m_soa.m_ax  [i], m_soa.m_ay [i], m_soa.m_az [i],
                        m_soa.m_p1x [i], m_soa.m_p1y[i], m_soa.m_p1z[i],
                        m_soa.m_p1x [j], m_soa.m_p1y[j], m_soa.m_p1z[j],
                        m_soa.m_mass[j], this->EPSILON                   );
                }

                m_soa.m_vx[i] += ( m_soa.m_ax[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                m_soa.m_vy[i] += ( m_soa.m_ay[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                m_soa.m_vz[i] += ( m_soa.m_az[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );

                m_soa.m_p0x[i] = m_soa.m_p1x[i] + m_soa.m_vx[i] * this->m_delta_t;
                m_soa.m_p0y[i] = m_soa.m_p1y[i] + m_soa.m_vy[i] * this->m_delta_t;
                m_soa.m_p0z[i] = m_soa.m_p1z[i] + m_soa.m_vz[i] * this->m_delta_t;
            }
            m_soa.m_p0_is_active = true;
        }
    }
};


template<class T>
class TestCaseNBody_SOA_NEON : public TestCaseNBody_baselineSOA<T> {


  protected:
    const size_t m_factor_loop_unrolling;

    inline float32x4_t sqrt_f32( float32x4_t v ) {

        float32x4_t rough    = vrsqrteq_f32( v );
        float32x4_t refined1 = vmulq_f32( vrsqrtsq_f32( vmulq_f32( rough, rough ), v ), rough );
        float32x4_t refined2 = vmulq_f32( vrsqrtsq_f32( vmulq_f32( refined1, refined1 ), v ), refined1 );
        // float32x4_t refined3 = vmulq_f32( vrsqrtsq_f32( vmulq_f32( refined2, refined2 ), v ), refined2 );
        return refined2;

    }

    virtual void inline bodyBodyInteraction_neon(

        T& a0x,       T& a0y,       T& a0z,
        const T p0x, const T p0y, const T p0z,
        const T* p1x, const T* p1y, const T* p1z,
        const T* mass1,
        const float32x4_t& qw_epsilon
    ) {
        const float32x4_t qw_p0x = { p0x, p0x, p0x, p0x };
        const float32x4_t qw_p0y = { p0y, p0y, p0y, p0y };
        const float32x4_t qw_p0z = { p0z, p0z, p0z, p0z };

        const float32x4_t qw_p1x = vld1q_f32( p1x );
        const float32x4_t qw_p1y = vld1q_f32( p1y );
        const float32x4_t qw_p1z = vld1q_f32( p1z );

        const float32x4_t qw_mass1 = vld1q_f32( mass1 );

        const float32x4_t qw_dx = vsubq_f32( qw_p1x, qw_p0x );
        const float32x4_t qw_dy = vsubq_f32( qw_p1y, qw_p0y );
        const float32x4_t qw_dz = vsubq_f32( qw_p1z, qw_p0z );

        const float32x4_t qw_dxdx = vmulq_f32( qw_dx, qw_dx );
        const float32x4_t qw_dydy = vmulq_f32( qw_dy, qw_dy );
        const float32x4_t qw_dzdz = vmulq_f32( qw_dz, qw_dz );

        const float32x4_t qw_subsum_1 = vaddq_f32( qw_dxdx, qw_dydy );
        const float32x4_t qw_subsum_2 = vaddq_f32( qw_dzdz, qw_epsilon );

        const float32x4_t qw_dist_sqr = vaddq_f32( qw_subsum_1, qw_subsum_2 );
        const float32x4_t qw_inv_dist = sqrt_f32( qw_dist_sqr );
        const float32x4_t qw_inv_dist_cube = vmulq_f32( vmulq_f32( qw_inv_dist, qw_inv_dist ), qw_inv_dist ); 
        const float32x4_t qw_s = vmulq_f32( qw_mass1, qw_inv_dist_cube );

        const float32x4_t qw_dxs = vmulq_f32( qw_dx, qw_s );
        const float32x4_t qw_dys = vmulq_f32( qw_dy, qw_s );
        const float32x4_t qw_dzs = vmulq_f32( qw_dz, qw_s );

        a0x += ( qw_dxs[0] + qw_dxs[1] + qw_dxs[2] + qw_dxs[3] );
        a0y += ( qw_dys[0] + qw_dys[1] + qw_dys[2] + qw_dys[3] );
        a0z += ( qw_dzs[0] + qw_dzs[1] + qw_dzs[2] + qw_dzs[3] );
    }

  public:

    TestCaseNBody_SOA_NEON( const size_t num_elements, const size_t factor_loop_unrolling, const T delta_t, const T tolerance )
        :TestCaseNBody_baselineSOA<T>( num_elements, delta_t, tolerance )
        ,m_factor_loop_unrolling( factor_loop_unrolling )
    {
        this->setNEON( 1, factor_loop_unrolling );
    }

    virtual ~TestCaseNBody_SOA_NEON(){;}

    virtual void run() {


        // reset the velocities.
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
            NBodyElem<T> e;            
            this->m_soa.get(e, i);
            e.m_v = this->m_v_saved[i].m_v;
            this->m_soa.set(e, i);
        }

        memset( this->m_soa.m_ax, 0, sizeof(float)*this->m_num_elements ); 
        memset( this->m_soa.m_ay, 0, sizeof(float)*this->m_num_elements ); 
        memset( this->m_soa.m_az, 0, sizeof(float)*this->m_num_elements ); 

        calc_block( 0, this->m_num_elements );
    }

    virtual void inline bodyBodyInteractionGuarded4LanesP0IsActive( int i, int j, const float32x4_t& qw_epsilon ) {

        if ( i < j || j+3 < i ) {
            bodyBodyInteraction_neon(
                this->m_soa.m_ax    [i  ],  this->m_soa.m_ay   [i  ],  this->m_soa.m_az   [i  ],
                this->m_soa.m_p0x   [i  ],  this->m_soa.m_p0y  [i  ],  this->m_soa.m_p0z  [i  ],
                &(this->m_soa.m_p0x [j  ]), &(this->m_soa.m_p0y[j  ]), &(this->m_soa.m_p0z[j  ]),
                &(this->m_soa.m_mass[j  ]), qw_epsilon           );
        }

        else {
            if ( i != j ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p0x [i  ], this->m_soa.m_p0y[i  ], this->m_soa.m_p0z[i  ],
                    this->m_soa.m_p0x [j  ], this->m_soa.m_p0y[j  ], this->m_soa.m_p0z[j  ],
                    this->m_soa.m_mass[j  ], this->EPSILON                      );
            }
            if ( i != j+1 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p0x [i  ], this->m_soa.m_p0y[i  ], this->m_soa.m_p0z[i  ],
                    this->m_soa.m_p0x [j+1], this->m_soa.m_p0y[j+1], this->m_soa.m_p0z[j+1],
                    this->m_soa.m_mass[j+1], this->EPSILON                      );
            }
            if ( i != j+2 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p0x [i  ], this->m_soa.m_p0y[i  ], this->m_soa.m_p0z[i  ],
                    this->m_soa.m_p0x [j+2], this->m_soa.m_p0y[j+2], this->m_soa.m_p0z[j+2],
                    this->m_soa.m_mass[j+2], this->EPSILON                      );
            }
            if ( i != j+3 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p0x [i  ], this->m_soa.m_p0y[i  ], this->m_soa.m_p0z[i  ],
                    this->m_soa.m_p0x [j+3], this->m_soa.m_p0y[j+3], this->m_soa.m_p0z[j+3],
                    this->m_soa.m_mass[j+3], this->EPSILON                      );
            }
        }
    }

    virtual void inline bodyBodyInteractionGuarded4LanesP1IsActive( int i, int j, const float32x4_t& qw_epsilon ) {

        if ( i < j || j+3 < i ) {
            bodyBodyInteraction_neon(
                this->m_soa.m_ax    [i  ],  this->m_soa.m_ay   [i  ],  this->m_soa.m_az   [i  ],
                this->m_soa.m_p1x   [i  ],  this->m_soa.m_p1y  [i  ],  this->m_soa.m_p1z  [i  ],
                &(this->m_soa.m_p1x [j  ]), &(this->m_soa.m_p1y[j  ]), &(this->m_soa.m_p1z[j  ]),
                &(this->m_soa.m_mass[j  ]), qw_epsilon           );
        }
        else {
            if ( i != j ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p1x [i  ], this->m_soa.m_p1y[i  ], this->m_soa.m_p1z[i  ],
                    this->m_soa.m_p1x [j  ], this->m_soa.m_p1y[j  ], this->m_soa.m_p1z[j  ],
                    this->m_soa.m_mass[j  ], this->EPSILON                      );
            }
            if ( i != j+1 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p1x [i  ], this->m_soa.m_p1y[i  ], this->m_soa.m_p1z[i  ],
                    this->m_soa.m_p1x [j+1], this->m_soa.m_p1y[j+1], this->m_soa.m_p1z[j+1],
                    this->m_soa.m_mass[j+1], this->EPSILON                      );
            }
            if ( i != j+2 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p1x [i  ], this->m_soa.m_p1y[i  ], this->m_soa.m_p1z[i  ],
                    this->m_soa.m_p1x [j+2], this->m_soa.m_p1y[j+2], this->m_soa.m_p1z[j+2],
                    this->m_soa.m_mass[j+2], this->EPSILON                      );
            }
            if ( i != j+3 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p1x [i  ], this->m_soa.m_p1y[i  ], this->m_soa.m_p1z[i  ],
                    this->m_soa.m_p1x [j+3], this->m_soa.m_p1y[j+3], this->m_soa.m_p1z[j+3],
                    this->m_soa.m_mass[j+3], this->EPSILON                      );
            }
        }
    }

    virtual void calc_block( const int elem_begin, const int elem_end_past_one )
    {
        const float32x4_t qw_epsilon{
            this->EPSILON, 
            this->EPSILON, 
            this->EPSILON, 
            this->EPSILON
        }; // used by bodyBodyInteraction_neon()

        if ( this->m_soa.m_p0_is_active ) { // take out 'if' out of the loop.

            for ( int i = elem_begin; i < elem_end_past_one ; i++ ) {

                if ( m_factor_loop_unrolling == 1 ) {

                    for ( int j = 0; j < this->m_num_elements ; j += 4 ) {

                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 2 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 8 ) {

                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j,   qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+4, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 4 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 16 ) {

                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j,    qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+ 4, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+ 8, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+12, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 8 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 32 ) {

                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j,    qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+ 4, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+ 8, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+12, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+16, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+20, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+24, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+28, qw_epsilon );
                    }
                }

                this->m_soa.m_vx[i]   += ( this->m_soa.m_ax[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                this->m_soa.m_vy[i]   += ( this->m_soa.m_ay[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                this->m_soa.m_vz[i]   += ( this->m_soa.m_az[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );

                this->m_soa.m_p1x[i] = this->m_soa.m_p0x[i] + this->m_soa.m_vx[i] * this->m_delta_t;
                this->m_soa.m_p1y[i] = this->m_soa.m_p0y[i] + this->m_soa.m_vy[i] * this->m_delta_t;
                this->m_soa.m_p1z[i] = this->m_soa.m_p0z[i] + this->m_soa.m_vz[i] * this->m_delta_t;
            }
            // this->soa_.p0_is_active_ = false;
        }
        else {
            assert(true);// this should never be called in the test cases.

            for ( size_t i = elem_begin; i < elem_end_past_one ; i++ ) {

                if ( m_factor_loop_unrolling == 1 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 4 ) {

                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 2 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 8 ) {

                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j,   qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+4, qw_epsilon );
                    }

                }
                else if ( m_factor_loop_unrolling == 4 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 16 ) {

                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j,    qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+ 4, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+ 8, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+12, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 8 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 32 ) {

                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j,    qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+ 4, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+ 8, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+12, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+16, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+20, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+24, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+28, qw_epsilon );
                    }
                }

                this->m_soa.m_vx[i]   += ( this->m_soa.m_ax[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                this->m_soa.m_vy[i]   += ( this->m_soa.m_ay[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                this->m_soa.m_vz[i]   += ( this->m_soa.m_az[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );

                this->m_soa.m_p1x[i] = this->m_soa.m_p0x[i] + this->m_soa.m_vx[i] * this->m_delta_t;
                this->m_soa.m_p1y[i] = this->m_soa.m_p0y[i] + this->m_soa.m_vy[i] * this->m_delta_t;
                this->m_soa.m_p1z[i] = this->m_soa.m_p0z[i] + this->m_soa.m_vz[i] * this->m_delta_t;
            }
            this->m_soa.m_p0_is_active = true;
        }
    }
};


template<class T>
class TestCaseNBody_neon_multithread_block : public TestCaseNBody_SOA_NEON<T> {

    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    const int                   m_num_threads;
    vector<thread>              m_threads;

  public:
    TestCaseNBody_neon_multithread_block( const size_t num_elements, const size_t factor_loop_unrolling, const int num_threads, const T delta_t, const T tolerance )
        :TestCaseNBody_SOA_NEON<T>( num_elements, factor_loop_unrolling, delta_t, tolerance )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
        ,m_num_threads( num_threads )
    {
        this->setNEON( num_threads, factor_loop_unrolling );

        const size_t num_elems_per_thread = this->m_num_elements / m_num_threads;

        auto thread_lambda = [ this, num_elems_per_thread ]( const size_t thread_index ) {

            const size_t elem_begin = thread_index * num_elems_per_thread;
            const size_t elem_end   = elem_begin + num_elems_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                this->calc_block( elem_begin, elem_end );

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

    virtual ~TestCaseNBody_neon_multithread_block(){

        m_fan_out.terminate();
        m_fan_in.terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    void run(){

        // reset the velocities.
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

            NBodyElem<T> e;            
            this->m_soa.get( e, i );
            e.m_v = this->m_v_saved[i].m_v;
            this->m_soa.set( e, i );
        }

        memset( this->m_soa.m_ax, 0, sizeof(T)*this->m_num_elements ); 
        memset( this->m_soa.m_ay, 0, sizeof(T)*this->m_num_elements ); 
        memset( this->m_soa.m_az, 0, sizeof(T)*this->m_num_elements ); 

        m_fan_out.notify();

        m_fan_in.wait();
    }
};


template<class T>
class TestCaseNBody_baselineAOS : public TestCaseNBody<T> {

    NBodyElem<T>*    m_aos;
    VelocityElem<T>* m_v_saved;
    bool             m_p0_is_active;

  public:
    virtual void inline bodyBodyInteraction_P0toP1( NBodyElem<T>& particle_i, const NBodyElem<T>& particle_j ) {

        const T dx = particle_j.m_p0.x - particle_i.m_p0.x;
        const T dy = particle_j.m_p0.y - particle_i.m_p0.y;
        const T dz = particle_j.m_p0.z - particle_i.m_p0.z;

        const T dist_sqr = dx*dx + dy*dy + dz*dz + this->EPSILON; 

        T inv_dist;

        // vDSP's rsqrt. No noticeable difference in speed.
        // const int num_1 = 1;
        //vvrsqrtf( &inv_dist, &dist_sqr, &num_1 ); 

        inv_dist = 1.0 / sqrtf( dist_sqr);

        const T inv_dist_cube = inv_dist * inv_dist * inv_dist;
        const T s = particle_j.m_am.w * inv_dist_cube;

        particle_i.m_am.x += (dx * s);
        particle_i.m_am.y += (dy * s);
        particle_i.m_am.z += (dz * s);
    }

    virtual void inline bodyBodyInteraction_P1toP0( NBodyElem<T>& particle_i, const NBodyElem<T>& particle_j ) {

        const T dx = particle_j.m_p1.x - particle_i.m_p1.x;
        const T dy = particle_j.m_p1.y - particle_i.m_p1.y;
        const T dz = particle_j.m_p1.z - particle_i.m_p1.z;

        const T dist_sqr = dx*dx + dy*dy + dz*dz + this->EPSILON; 

        T inv_dist;

        // vDSP's rsqrt. No noticeable difference in speed.
        // const int num_1 = 1;
        //vvrsqrtf( &inv_dist, &dist_sqr, &num_1 ); 

        inv_dist = 1.0 / sqrtf( dist_sqr);

        const T inv_dist_cube = inv_dist * inv_dist * inv_dist;
        const T s = particle_j.m_am.w * inv_dist_cube;

        particle_i.m_am.x += (dx * s);
        particle_i.m_am.y += (dy * s);
        particle_i.m_am.z += (dz * s);
    }


  public:

    TestCaseNBody_baselineAOS( const size_t num_elements, const T delta_t, const T tolerance )
        :TestCaseNBody<T>( num_elements, delta_t, tolerance )
        ,m_aos         ( new NBodyElem<T>   [num_elements] )
        ,m_v_saved     ( new VelocityElem<T>[num_elements] )
        ,m_p0_is_active( true )
    {
        this->setAOS     ( num_elements );
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseNBody_baselineAOS() {

        delete[] m_aos;
        delete[] m_v_saved;
    }

    virtual void setInitialStates( const NBodyElem<T>* const src_array ) {

        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

             m_aos[i]     = src_array[i];
             m_v_saved[i] = src_array[i].m_v;
        }
    }

    virtual NBodyElem<T> getParticleAt( const size_t i ) {

        return m_aos[i];
    }

    virtual void run() {

        // reset the velocity after every iteration.
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
             m_aos[i].m_v = m_v_saved[i].m_v;
        }

        if ( m_p0_is_active ) { // take out 'if' out of the loop.

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                auto& particle_i = m_aos[i];

                particle_i.m_am.x = 0.0;
                particle_i.m_am.y = 0.0;
                particle_i.m_am.z = 0.0;

                for ( size_t j = 0; j < this->m_num_elements ; j ++ ) {
                    if ( i != j ) {
                        const auto& particle_j = m_aos[j];
                   
                        bodyBodyInteraction_P0toP1( particle_i, particle_j );
                    }
                }

                particle_i.m_v.x += ( particle_i.m_am.x * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );
                particle_i.m_v.y += ( particle_i.m_am.y * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );
                particle_i.m_v.z += ( particle_i.m_am.z * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );

                particle_i.m_p1.x = particle_i.m_p0.x + particle_i.m_v.x * this->m_delta_t;
                particle_i.m_p1.y = particle_i.m_p0.y + particle_i.m_v.y * this->m_delta_t;
                particle_i.m_p1.z = particle_i.m_p0.z + particle_i.m_v.z * this->m_delta_t;
            }

            // m_p0_is_active = false;
        }
        else {
            assert(true);// this should never be called in the tests.

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                auto& particle_i = m_aos[i];

                particle_i.m_am.x = 0.0;
                particle_i.m_am.y = 0.0;
                particle_i.m_am.z = 0.0;

                for ( size_t j = 0; j < this->m_num_elements ; j ++ ) {
                    if ( i != j ) {
                        const auto& particle_j = m_aos[j];

                        bodyBodyInteraction_P1toP0( particle_i, particle_j );
                     }
                }

                particle_i.m_v.x += ( particle_i.m_am.x * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );
                particle_i.m_v.y += ( particle_i.m_am.y * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );
                particle_i.m_v.z += ( particle_i.m_am.z * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );

                particle_i.m_p0.x = particle_i.m_p1.x + particle_i.m_v.x * this->m_delta_t;
                particle_i.m_p0.y = particle_i.m_p1.y + particle_i.m_v.y * this->m_delta_t;
                particle_i.m_p0.z = particle_i.m_p1.z + particle_i.m_v.z * this->m_delta_t;
            }
            m_p0_is_active = true;
        }
    }
};


template<class T>
class TestCaseNBody_Metal : public TestCaseNBody<T> {

    NBodyMetalCpp    m_metal;
    VelocityElem<T>* m_v_saved;
    bool             m_p0_is_active;

  public:

    TestCaseNBody_Metal( const size_t num_elements, const T delta_t, const T tolerance )
        :TestCaseNBody<T>( num_elements, delta_t, tolerance )
        ,m_metal       ( num_elements )
        ,m_v_saved     ( new VelocityElem<T>[num_elements] )
        ,m_p0_is_active( true )
    {
        this->setAOS( num_elements );

        int num_threads_per_group;
        int num_groups_per_grid;

        if ( num_elements <= 1024 ) {
            num_threads_per_group = ((num_elements + 31)/32) * 32;
            num_groups_per_grid = 1;
        }
        else {
            num_threads_per_group = 1024;
            num_groups_per_grid = (num_elements + 1023) / 1024;
        }
        this->setMetal( DEFAULT, num_groups_per_grid, num_threads_per_group );
    }

    virtual ~TestCaseNBody_Metal() {
        delete[] m_v_saved;
    }

    virtual void setInitialStates( const NBodyElem<T>* const src_array ) {
        memcpy( m_metal.getRawPointerParticles(), src_array, sizeof(struct particle) * (this->m_num_elements) );

        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
             m_v_saved[i] = src_array[i].m_v;
        }
    }

    virtual NBodyElem<T> getParticleAt( const size_t i ) {
        return ((NBodyElem<T>*)m_metal.getRawPointerParticles())[i];
    }

    virtual void run() {
        
        // resete the velocity after every iteration.
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

            auto* p = (NBodyElem<T>*)m_metal.getRawPointerParticles();
            p[i].m_v = m_v_saved[i].m_v;
        }

        if ( m_p0_is_active ) { // take out 'if' out of the loop.

            m_metal.performComputationDirectionIsP0ToP1( true );
            // m_p0_is_active = false;
        }
        else {

            assert(true);// this should never be called in the tests.
            m_metal.performComputationDirectionIsP0ToP1( false );

            m_p0_is_active = true;
        }
    }
};


template <class T>
class TestExecutorNBody : public TestExecutor {

  protected:

    const int             m_num_elements;
    const bool            m_repeatable;
    default_random_engine m_e;

    NBodyElem<T>*         m_particles;
    NBodyElem<T>*         m_particles_baseline;

  public:
    TestExecutorNBody( ostream& os, const int num_elements, const int num_trials, const bool repeatable )
        :TestExecutor        ( os, num_trials )
        ,m_num_elements      ( num_elements )
        ,m_repeatable        ( repeatable )
        ,m_e                 ( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_particles         ( nullptr )
        ,m_particles_baseline( nullptr )
    {
        m_particles = new NBodyElem<T>[ num_elements ];

        for ( size_t i = 0 ; i < num_elements; i++ ) {
            auto& p = m_particles[i];
            p.setRandomInitialState( m_e );
        }

        m_particles_baseline = new NBodyElem<T>[ num_elements ];
    }

    virtual ~TestExecutorNBody() {

        delete[] m_particles;
        delete[] m_particles_baseline;

    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCaseNBody<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            for ( int j = 0; j < m_num_elements; j++ ) {

                m_particles_baseline[j] = t->getParticleAt(j);
            }
        }

        t->compareTruth( m_particles_baseline );
    }

    void prepareForRun ( const int test_case, const int num ) {

        auto t = dynamic_pointer_cast< TestCaseNBody<T> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_particles );
    }
};

static const size_t NUM_TRIALS = 10;
static const float  TIMESTEP   = 0.1;
static const float  TOLERANCE  = 0.01;

size_t nums_elements[]{ 32, 64, 128, 256, 512, 1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024 };

template<class T>
void testSuitePerType ( const T delta_t, const T tolerance ) {

    const int neon_num_lanes = ( is_same<float, T>::value )? 4 : 2;

    for( auto num_elements : nums_elements ) {

        TestExecutorNBody<T> e( cout, num_elements, NUM_TRIALS, false );

        e.addTestCase( make_shared< TestCaseNBody_baselineAOS <T> > ( num_elements, delta_t, tolerance ) );
        e.addTestCase( make_shared< TestCaseNBody_baselineSOA <T> > ( num_elements, delta_t, tolerance ) );
        e.addTestCase( make_shared< TestCaseNBody_SOA_NEON <T> > ( num_elements, 1, delta_t, tolerance ) );
        e.addTestCase( make_shared< TestCaseNBody_SOA_NEON <T> > ( num_elements, 2, delta_t, tolerance ) );
        if ( num_elements >= 4 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseNBody_SOA_NEON <T> > ( num_elements, 4, delta_t, tolerance ) );
        }
        if ( num_elements >= 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseNBody_SOA_NEON <T> > ( num_elements, 8, delta_t, tolerance ) );
        }
        if ( num_elements >= 1 * 2 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseNBody_neon_multithread_block <T> > ( num_elements, 1, 2, delta_t, tolerance ) );
        }
        if ( num_elements >= 1 * 4 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseNBody_neon_multithread_block <T> > ( num_elements, 1, 4, delta_t, tolerance ) );
        }
        if ( num_elements >= 1 * 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseNBody_neon_multithread_block <T> > ( num_elements, 1, 8, delta_t, tolerance ) );
        }
        e.addTestCase( make_shared< TestCaseNBody_Metal <T> > ( num_elements, delta_t, tolerance ) );

        e.execute();
    }
}
#if TARGET_OS_OSX
int main( int argc, char* argv[] ) {
#else
int run_test() {
#endif
    TestCaseWithTimeMeasurements::printHeader( cout );

    cerr << "\n\nTesting for type float.\n\n";

    testSuitePerType<float> ( TIMESTEP, TOLERANCE );

    return 0;
}
