#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <Accelerate/Accelerate.h>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"
#include "saxpy_metal_cpp.h"

template<class T>
class TestCaseSAXPY :public TestCaseWithTimeMeasurements {

  protected:
    const size_t m_num_elements;
    const T*     m_x;
    T*           m_y;
    T            m_alpha;

  public:
    TestCaseSAXPY( const size_t num_elements )
        :m_num_elements( num_elements )
        ,m_x           ( nullptr      )
        ,m_y           ( nullptr      )
        ,m_alpha       ( 0.0          )
    {
        if constexpr ( is_same<float, T>::value ) {
            setDataElementType( FLOAT );
        }
        else if constexpr ( is_same<double, T>::value ) {
            setDataElementType( DOUBLE );
        }
        else if constexpr ( is_same<int, T>::value ) {
            setDataElementType( INT );
        }
        setVector( num_elements );
        setVerificationType(RMS);
    }

    virtual ~TestCaseSAXPY(){;}

    virtual bool needsToCopyBackY() = 0;

    void calculateRMS( const T* baseline ) {
         auto rms = getRMSDiffTwoVectors( baseline, getY(), m_num_elements) ;
         setRMS(rms);
    }

    virtual void setX    ( const T* const x ){ m_x = x; }
    virtual void setY    ( T* const y       ){ m_y = y; }
    virtual void setAlpha( const T a        ){ m_alpha = a; }
    virtual T*   getY    ()                  { return m_y; }

    virtual void run() = 0;
};


template<class T>
class TestCaseSAXPY_baseline : public TestCaseSAXPY<T> {

  public:
    TestCaseSAXPY_baseline( const size_t num_elements )
        :TestCaseSAXPY<T>( num_elements )
    {
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseSAXPY_baseline(){;}

    virtual bool needsToCopyBackY() { return false; }

    void run() {
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
            this->m_y[i] = this->m_alpha * this->m_x[i] + this->m_y[i];
        }
    }
};

template<class T>
class TestCaseSAXPY_neon : public TestCaseSAXPY<T> {

  protected:
    const size_t m_factor_loop_unrolling;

    void (TestCaseSAXPY_neon<T>::*m_calc_block)(const int, const int);

    void calc_block_factor_1(const int elem_begin, const int elem_end_past_one) {

        if constexpr ( is_same<float, T>::value ) {

            for (size_t i = elem_begin;  i < elem_end_past_one; i += 4 ) {

                //__builtin_prefetch(&this->m_x[i+4], 0 );
                //__builtin_prefetch(&this->m_y[i+4], 1 );

                const float32x4_t x_quad    = vld1q_f32( &this->m_x[i] );
                const float32x4_t ax_quad   = vmulq_n_f32( x_quad, this->m_alpha );
                const float32x4_t y_quad    = vld1q_f32( &this->m_y[i] );
                const float32x4_t axpy_quad = vaddq_f32( ax_quad, y_quad );
                vst1q_f32( &(this->m_y[i]), axpy_quad );
            }
        }      
        else {
            for (size_t i = elem_begin;  i < elem_end_past_one; i += 2 ) {

                //__builtin_prefetch(&this->m_x[i+2], 0 );
                //__builtin_prefetch(&this->m_y[i+2], 1 );

                const float64x2_t x_pair    = vld1q_f64( &this->m_x[i] );
                const float64x2_t ax_pair   = vmulq_n_f64( x_pair, this->m_alpha );
                const float64x2_t y_pair    = vld1q_f64( &this->m_y[i] );
                const float64x2_t axpy_pair = vaddq_f64( ax_pair, y_pair );
                vst1q_f64( &(this->m_y[i]), axpy_pair );
            }
        }
    }


    void calc_block_factor_2(const int elem_begin, const int elem_end_past_one) {

        if constexpr ( is_same<float, T>::value ) {

            for (size_t i = elem_begin;  i < elem_end_past_one; i += 8 ) {

                //__builtin_prefetch(&this->m_x[i+8], 0 );
                //__builtin_prefetch(&this->m_y[i+8], 1 );

                const float32x4_t x_quad1    = vld1q_f32( &this->m_x[i] );
                const float32x4_t x_quad2    = vld1q_f32( &this->m_x[i+4] );
                const float32x4_t y_quad1    = vld1q_f32( &this->m_y[i] );
                const float32x4_t y_quad2    = vld1q_f32( &this->m_y[i+4] );
                const float32x4_t ax_quad1   = vmulq_n_f32( x_quad1, this->m_alpha );
                const float32x4_t ax_quad2   = vmulq_n_f32( x_quad2, this->m_alpha );
                const float32x4_t axpy_quad1 = vaddq_f32( ax_quad1, y_quad1 );
                const float32x4_t axpy_quad2 = vaddq_f32( ax_quad2, y_quad2 );
                vst1q_f32( &(this->m_y[i]),    axpy_quad1 );
                vst1q_f32( &(this->m_y[i+4]),  axpy_quad2 );
            }
        }      
        else {
            for (size_t i = elem_begin;  i < elem_end_past_one; i += 4 ) {

                //__builtin_prefetch(&this->m_x[i+4], 0 );
                //__builtin_prefetch(&this->m_y[i+4], 1 );

                const float64x2_t x_pair1    = vld1q_f64( &this->m_x[i  ] );
                const float64x2_t x_pair2    = vld1q_f64( &this->m_x[i+2] );
                const float64x2_t y_pair1    = vld1q_f64( &this->m_y[i  ] );
                const float64x2_t y_pair2    = vld1q_f64( &this->m_y[i+2] );
                const float64x2_t ax_pair1   = vmulq_n_f64( x_pair1, this->m_alpha );
                const float64x2_t ax_pair2   = vmulq_n_f64( x_pair2, this->m_alpha );
                const float64x2_t axpy_pair1 = vaddq_f64( ax_pair1, y_pair1 );
                const float64x2_t axpy_pair2 = vaddq_f64( ax_pair2, y_pair2 );
                vst1q_f64( &(this->m_y[i  ]), axpy_pair1 );
                vst1q_f64( &(this->m_y[i+2]), axpy_pair2 );
            }
        }
    }


    void calc_block_factor_4(const int elem_begin, const int elem_end_past_one) {

        if constexpr ( is_same<float, T>::value ) {

            for (size_t i = elem_begin;  i < elem_end_past_one; i += 16 ) {

                //__builtin_prefetch(&this->m_x[i+16], 0 );
                //__builtin_prefetch(&this->m_y[i+16], 1 );

                const float32x4_t x_quad1    = vld1q_f32( &this->m_x[i   ] );
                const float32x4_t x_quad2    = vld1q_f32( &this->m_x[i+ 4] );
                const float32x4_t x_quad3    = vld1q_f32( &this->m_x[i+ 8] );
                const float32x4_t x_quad4    = vld1q_f32( &this->m_x[i+12] );
                const float32x4_t y_quad1    = vld1q_f32( &this->m_y[i   ] );
                const float32x4_t y_quad2    = vld1q_f32( &this->m_y[i+ 4] );
                const float32x4_t y_quad3    = vld1q_f32( &this->m_y[i+ 8] );
                const float32x4_t y_quad4    = vld1q_f32( &this->m_y[i+12] );
                const float32x4_t ax_quad1   = vmulq_n_f32( x_quad1, this->m_alpha );
                const float32x4_t ax_quad2   = vmulq_n_f32( x_quad2, this->m_alpha );
                const float32x4_t ax_quad3   = vmulq_n_f32( x_quad3, this->m_alpha );
                const float32x4_t ax_quad4   = vmulq_n_f32( x_quad4, this->m_alpha );
                const float32x4_t axpy_quad1 = vaddq_f32( ax_quad1, y_quad1 );
                const float32x4_t axpy_quad2 = vaddq_f32( ax_quad2, y_quad2 );
                const float32x4_t axpy_quad3 = vaddq_f32( ax_quad3, y_quad3 );
                const float32x4_t axpy_quad4 = vaddq_f32( ax_quad4, y_quad4 );
                vst1q_f32( &(this->m_y[i   ]), axpy_quad1 );
                vst1q_f32( &(this->m_y[i+ 4]), axpy_quad2 );
                vst1q_f32( &(this->m_y[i+ 8]), axpy_quad3 );
                vst1q_f32( &(this->m_y[i+12]), axpy_quad4 );
            }
        }      
        else {
            for (size_t i = elem_begin;  i < elem_end_past_one; i += 8 ) {

                //__builtin_prefetch(&this->m_x[i+8], 0 );
                //__builtin_prefetch(&this->m_y[i+8], 1 );

                const float64x2_t x_pair1    = vld1q_f64( &this->m_x[i  ] );
                const float64x2_t x_pair2    = vld1q_f64( &this->m_x[i+2] );
                const float64x2_t x_pair3    = vld1q_f64( &this->m_x[i+4] );
                const float64x2_t x_pair4    = vld1q_f64( &this->m_x[i+6] );
                const float64x2_t y_pair1    = vld1q_f64( &this->m_y[i  ] );
                const float64x2_t y_pair2    = vld1q_f64( &this->m_y[i+2] );
                const float64x2_t y_pair3    = vld1q_f64( &this->m_y[i+4] );
                const float64x2_t y_pair4    = vld1q_f64( &this->m_y[i+6] );
                const float64x2_t ax_pair1   = vmulq_n_f64( x_pair1, this->m_alpha );
                const float64x2_t ax_pair2   = vmulq_n_f64( x_pair2, this->m_alpha );
                const float64x2_t ax_pair3   = vmulq_n_f64( x_pair3, this->m_alpha );
                const float64x2_t ax_pair4   = vmulq_n_f64( x_pair4, this->m_alpha );
                const float64x2_t axpy_pair1 = vaddq_f64( ax_pair1, y_pair1 );
                const float64x2_t axpy_pair2 = vaddq_f64( ax_pair2, y_pair2 );
                const float64x2_t axpy_pair3 = vaddq_f64( ax_pair3, y_pair3 );
                const float64x2_t axpy_pair4 = vaddq_f64( ax_pair4, y_pair4 );
                vst1q_f64( &(this->m_y[i  ]), axpy_pair1 );
                vst1q_f64( &(this->m_y[i+2]), axpy_pair2 );
                vst1q_f64( &(this->m_y[i+4]), axpy_pair3 );
                vst1q_f64( &(this->m_y[i+6]), axpy_pair4 );
            }
        }
    }


    void calc_block_factor_8(const int elem_begin, const int elem_end_past_one) {

        if constexpr ( is_same<float, T>::value ) {

            for (size_t i = elem_begin;  i < elem_end_past_one; i += 32 ) {

                //__builtin_prefetch(&this->m_x[i+32], 0 );
                //__builtin_prefetch(&this->m_y[i+32], 1 );

                const float32x4_t x_quad1    = vld1q_f32( &this->m_x[i   ] );
                const float32x4_t x_quad2    = vld1q_f32( &this->m_x[i+ 4] );
                const float32x4_t x_quad3    = vld1q_f32( &this->m_x[i+ 8] );
                const float32x4_t x_quad4    = vld1q_f32( &this->m_x[i+12] );
                const float32x4_t x_quad5    = vld1q_f32( &this->m_x[i+16] );
                const float32x4_t x_quad6    = vld1q_f32( &this->m_x[i+20] );
                const float32x4_t x_quad7    = vld1q_f32( &this->m_x[i+24] );
                const float32x4_t x_quad8    = vld1q_f32( &this->m_x[i+28] );
                const float32x4_t y_quad1    = vld1q_f32( &this->m_y[i   ] );
                const float32x4_t y_quad2    = vld1q_f32( &this->m_y[i+ 4] );
                const float32x4_t y_quad3    = vld1q_f32( &this->m_y[i+ 8] );
                const float32x4_t y_quad4    = vld1q_f32( &this->m_y[i+12] );
                const float32x4_t y_quad5    = vld1q_f32( &this->m_y[i+16] );
                const float32x4_t y_quad6    = vld1q_f32( &this->m_y[i+20] );
                const float32x4_t y_quad7    = vld1q_f32( &this->m_y[i+24] );
                const float32x4_t y_quad8    = vld1q_f32( &this->m_y[i+28] );
                const float32x4_t ax_quad1   = vmulq_n_f32( x_quad1, this->m_alpha );
                const float32x4_t ax_quad2   = vmulq_n_f32( x_quad2, this->m_alpha );
                const float32x4_t ax_quad3   = vmulq_n_f32( x_quad3, this->m_alpha );
                const float32x4_t ax_quad4   = vmulq_n_f32( x_quad4, this->m_alpha );
                const float32x4_t ax_quad5   = vmulq_n_f32( x_quad5, this->m_alpha );
                const float32x4_t ax_quad6   = vmulq_n_f32( x_quad6, this->m_alpha );
                const float32x4_t ax_quad7   = vmulq_n_f32( x_quad7, this->m_alpha );
                const float32x4_t ax_quad8   = vmulq_n_f32( x_quad8, this->m_alpha );
                const float32x4_t axpy_quad1 = vaddq_f32( ax_quad1, y_quad1 );
                const float32x4_t axpy_quad2 = vaddq_f32( ax_quad2, y_quad2 );
                const float32x4_t axpy_quad3 = vaddq_f32( ax_quad3, y_quad3 );
                const float32x4_t axpy_quad4 = vaddq_f32( ax_quad4, y_quad4 );
                const float32x4_t axpy_quad5 = vaddq_f32( ax_quad5, y_quad5 );
                const float32x4_t axpy_quad6 = vaddq_f32( ax_quad6, y_quad6 );
                const float32x4_t axpy_quad7 = vaddq_f32( ax_quad7, y_quad7 );
                const float32x4_t axpy_quad8 = vaddq_f32( ax_quad8, y_quad8 );
                vst1q_f32( &(this->m_y[i   ]), axpy_quad1 );
                vst1q_f32( &(this->m_y[i+ 4]), axpy_quad2 );
                vst1q_f32( &(this->m_y[i+ 8]), axpy_quad3 );
                vst1q_f32( &(this->m_y[i+12]), axpy_quad4 );
                vst1q_f32( &(this->m_y[i+16]), axpy_quad5 );
                vst1q_f32( &(this->m_y[i+20]), axpy_quad6 );
                vst1q_f32( &(this->m_y[i+24]), axpy_quad7 );
                vst1q_f32( &(this->m_y[i+28]), axpy_quad8 );
            }
        }      
        else {
            for (size_t i = elem_begin;  i < elem_end_past_one; i += 16 ) {

                //__builtin_prefetch(&this->m_x[i+16], 0 );
                //__builtin_prefetch(&this->m_y[i+16], 1 );

                const float64x2_t x_pair1    = vld1q_f64( &this->m_x[i   ] );
                const float64x2_t x_pair2    = vld1q_f64( &this->m_x[i+ 2] );
                const float64x2_t x_pair3    = vld1q_f64( &this->m_x[i+ 4] );
                const float64x2_t x_pair4    = vld1q_f64( &this->m_x[i+ 6] );
                const float64x2_t x_pair5    = vld1q_f64( &this->m_x[i+ 8] );
                const float64x2_t x_pair6    = vld1q_f64( &this->m_x[i+10] );
                const float64x2_t x_pair7    = vld1q_f64( &this->m_x[i+12] );
                const float64x2_t x_pair8    = vld1q_f64( &this->m_x[i+14] );
                const float64x2_t y_pair1    = vld1q_f64( &this->m_y[i   ] );
                const float64x2_t y_pair2    = vld1q_f64( &this->m_y[i+ 2] );
                const float64x2_t y_pair3    = vld1q_f64( &this->m_y[i+ 4] );
                const float64x2_t y_pair4    = vld1q_f64( &this->m_y[i+ 6] );
                const float64x2_t y_pair5    = vld1q_f64( &this->m_y[i+ 8] );
                const float64x2_t y_pair6    = vld1q_f64( &this->m_y[i+10] );
                const float64x2_t y_pair7    = vld1q_f64( &this->m_y[i+12] );
                const float64x2_t y_pair8    = vld1q_f64( &this->m_y[i+14] );
                const float64x2_t ax_pair1   = vmulq_n_f64( x_pair1, this->m_alpha );
                const float64x2_t ax_pair2   = vmulq_n_f64( x_pair2, this->m_alpha );
                const float64x2_t ax_pair3   = vmulq_n_f64( x_pair3, this->m_alpha );
                const float64x2_t ax_pair4   = vmulq_n_f64( x_pair4, this->m_alpha );
                const float64x2_t ax_pair5   = vmulq_n_f64( x_pair5, this->m_alpha );
                const float64x2_t ax_pair6   = vmulq_n_f64( x_pair6, this->m_alpha );
                const float64x2_t ax_pair7   = vmulq_n_f64( x_pair7, this->m_alpha );
                const float64x2_t ax_pair8   = vmulq_n_f64( x_pair8, this->m_alpha );
                const float64x2_t axpy_pair1 = vaddq_f64( ax_pair1, y_pair1 );
                const float64x2_t axpy_pair2 = vaddq_f64( ax_pair2, y_pair2 );
                const float64x2_t axpy_pair3 = vaddq_f64( ax_pair3, y_pair3 );
                const float64x2_t axpy_pair4 = vaddq_f64( ax_pair4, y_pair4 );
                const float64x2_t axpy_pair5 = vaddq_f64( ax_pair5, y_pair5 );
                const float64x2_t axpy_pair6 = vaddq_f64( ax_pair6, y_pair6 );
                const float64x2_t axpy_pair7 = vaddq_f64( ax_pair7, y_pair7 );
                const float64x2_t axpy_pair8 = vaddq_f64( ax_pair8, y_pair8 );
                vst1q_f64( &(this->m_y[i   ]), axpy_pair1 );
                vst1q_f64( &(this->m_y[i+ 2]), axpy_pair2 );
                vst1q_f64( &(this->m_y[i+ 4]), axpy_pair3 );
                vst1q_f64( &(this->m_y[i+ 6]), axpy_pair4 );
                vst1q_f64( &(this->m_y[i+ 8]), axpy_pair5 );
                vst1q_f64( &(this->m_y[i+10]), axpy_pair6 );
                vst1q_f64( &(this->m_y[i+12]), axpy_pair7 );
                vst1q_f64( &(this->m_y[i+14]), axpy_pair8 );
            }
        }
    }


  public:
    TestCaseSAXPY_neon( const size_t num_elements, const size_t factor_loop_unrolling )
        :TestCaseSAXPY<T>( num_elements )
        ,m_factor_loop_unrolling( factor_loop_unrolling )
    {
        this->setNEON( 1, factor_loop_unrolling );
        if (factor_loop_unrolling == 1) {
            m_calc_block = &TestCaseSAXPY_neon::calc_block_factor_1;
        }
        else if (factor_loop_unrolling == 2) {
            m_calc_block = &TestCaseSAXPY_neon::calc_block_factor_2;
        }
        else if (factor_loop_unrolling == 4) {
            m_calc_block = &TestCaseSAXPY_neon::calc_block_factor_4;
        }
        else if (factor_loop_unrolling == 8) {
            m_calc_block = &TestCaseSAXPY_neon::calc_block_factor_8;
        }
        else {
            assert(true);
        }
    }

    virtual ~TestCaseSAXPY_neon(){;}

    virtual bool needsToCopyBackY() { return false; }

    virtual inline void call_block( const int elem_begin, const int elem_end_past_one ) {
        (this->*m_calc_block)( elem_begin, elem_end_past_one );
    }

    void run() {
        call_block( 0, this->m_num_elements );
    }
};



template<class T>
class TestCaseSAXPY_neon_multithread_block : public TestCaseSAXPY_neon<T> {

    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    const int                   m_num_threads;
    vector<thread>              m_threads;

  public:
    TestCaseSAXPY_neon_multithread_block( const size_t num_elements, const size_t factor_loop_unrolling, const size_t num_threads )
        :TestCaseSAXPY_neon<T>( num_elements, factor_loop_unrolling )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
        ,m_num_threads( num_threads )
    {
        this->setNEON( num_threads, factor_loop_unrolling );

        const size_t num_elems_per_thread = this->m_num_elements / m_num_threads;

        auto thread_lambda = [this, num_elems_per_thread ]( const size_t thread_index ) {

            const size_t elem_begin = thread_index * num_elems_per_thread;
            const size_t elem_end   = elem_begin + num_elems_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }
                this->call_block( elem_begin, elem_end );

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

    virtual ~TestCaseSAXPY_neon_multithread_block(){

        m_fan_out.terminate();
        m_fan_in.terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    virtual bool needsToCopyBackY() { return false; }

    void run() {

        m_fan_out.notify();

        m_fan_in.wait();
    }
};


template<class T>
class TestCaseSAXPY_multithread_block : public TestCaseSAXPY<T> {

    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    const int                   m_num_threads;
    vector<thread>              m_threads;

  public:
    TestCaseSAXPY_multithread_block( const size_t num_elements, const int num_threads )
        :TestCaseSAXPY<T>( num_elements )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
        ,m_num_threads( num_threads )
    {
        this->setCPPBlock( num_threads, 1 );

        const size_t num_elems_per_thread = this->m_num_elements / m_num_threads;

        auto thread_lambda = [this, num_elems_per_thread ]( const size_t thread_index ) {

            const size_t elem_begin = thread_index * num_elems_per_thread;
            const size_t elem_end   = elem_begin + num_elems_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                for ( size_t i = elem_begin; i < elem_end; i++ ) {

                   this->m_y[i] = this->m_alpha * this->m_x[i] + this->m_y[i];
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

    virtual ~TestCaseSAXPY_multithread_block(){

        m_fan_out.terminate();
        m_fan_in.terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    virtual bool needsToCopyBackY() { return false; }

    void run() {

        m_fan_out.notify();

        m_fan_in.wait();
    }
};


template<class T>
class TestCaseSAXPY_multithread_interleave : public TestCaseSAXPY<T> {

    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    const int                   m_num_threads;
    vector<thread>              m_threads;

  public:
    TestCaseSAXPY_multithread_interleave( const size_t num_elements, const int num_threads )
        :TestCaseSAXPY<T>( num_elements )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
        ,m_num_threads( num_threads )
    {
        this->setCPPInterleaved( num_threads, 1 );

        auto thread_lambda = [this]( const size_t thread_index ) {

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                for ( size_t i = thread_index; i < this->m_num_elements ; i += m_num_threads ) {
                    this->m_y[i] = this->m_alpha * this->m_x[i] + this->m_y[i];
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

    virtual ~TestCaseSAXPY_multithread_interleave(){

        m_fan_out.terminate();
        m_fan_in.terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    virtual bool needsToCopyBackY() { return false; }

    void run() {

        m_fan_out.notify();

        m_fan_in.wait();
    }
};


template<class T>
class TestCaseSAXPY_vDSP : public TestCaseSAXPY<T> {

  public:
    TestCaseSAXPY_vDSP( const size_t m_num_elements )
        :TestCaseSAXPY<T>( m_num_elements )
    {
        static_assert( is_same<float, T>::value || is_same<double, T>::value );

        this->setImplementationType(VDSP);
    }
    virtual ~TestCaseSAXPY_vDSP(){;}

    virtual bool needsToCopyBackY() { return false; }

    void run();
};

template<>
void TestCaseSAXPY_vDSP<float>::run()
{
    vDSP_vsma ( this->m_x, 1, &(this->m_alpha), this->m_y, 1, this->m_y, 1, this->m_num_elements );
}

template<>
void TestCaseSAXPY_vDSP<double>::run()
{
    vDSP_vsmaD ( this->m_x, 1, &(this->m_alpha), this->m_y, 1, this->m_y, 1, this->m_num_elements );
}

template<class T>
class TestCaseSAXPY_BLAS : public TestCaseSAXPY<T> {
  public:
    TestCaseSAXPY_BLAS( const size_t num_elements )
        :TestCaseSAXPY<T>( num_elements )
    {
        static_assert( is_same<float, T>::value || is_same<double, T>::value );

        this->setImplementationType(BLAS);
    }
    virtual ~TestCaseSAXPY_BLAS(){;}

    virtual bool needsToCopyBackY() { return false; }

    void run();
};

template<>
void TestCaseSAXPY_BLAS<float>::run()
{
    cblas_saxpy( this->m_num_elements, this->m_alpha, this->m_x, 1, this->m_y, 1 );
}

template<>
void TestCaseSAXPY_BLAS<double>::run()
{
    cblas_daxpy( this->m_num_elements, this->m_alpha, this->m_x, 1, this->m_y, 1 );
}

template<class T>
class TestCaseSAXPY_Metal : public TestCaseSAXPY<T> {

  private:
    SaxpyMetalCpp m_metal;

  public:
    TestCaseSAXPY_Metal( const size_t num_elements , const size_t num_threads_per_group, const size_t num_groups_per_grid )
        :TestCaseSAXPY<T>( num_elements )
        ,m_metal( num_elements, num_threads_per_group, num_groups_per_grid )
    {
        static_assert( std::is_same<float, T>::value );

        this->setMetal( DEFAULT, num_groups_per_grid, num_threads_per_group );
    }

    virtual ~TestCaseSAXPY_Metal(){;}

    virtual bool needsToCopyBackY() { return true; }

    void setX( const T* const x ){
        memcpy( m_metal.getRawPointerX(), x, this->m_num_elements*sizeof(T) );
    }

    void setY( T* const y ){
        memcpy( m_metal.getRawPointerY(), y, this->m_num_elements*sizeof(T) );
    }

    void setAlpha( const T a ){ m_metal.setScalar_a( a ); }

    void run() {
        m_metal.performComputation();
    }

    T* getY(){ return m_metal.getRawPointerY(); }
};


template <class T>
class TestExecutorSAXPY : public TestExecutor {

  protected:

    const int             m_num_elements;
    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_x;
    T*                    m_y_org;
    T                     m_alpha;
    T*                    m_y_out;
    T*                    m_y_baseline;

  public:

    TestExecutorSAXPY(
        ostream&   os,
        const int  num_elements,
        const int  num_trials,
        const bool repeatable,
        const T    min_val,
        const T    max_val 
    )
        :TestExecutor  ( os, num_trials )
        ,m_num_elements( num_elements )
        ,m_repeatable  ( repeatable )
        ,m_e           ( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_x           ( new T[num_elements] )
        ,m_y_org       ( new T[num_elements] )
        ,m_alpha       ( 0.0 )
        ,m_y_out       ( new T[num_elements] )
        ,m_y_baseline  ( new T[num_elements] )
    {
        fillArrayWithRandomValues( m_e, m_x,     m_num_elements, min_val, max_val );
        fillArrayWithRandomValues( m_e, m_y_org, m_num_elements, min_val, max_val );

        m_alpha = getRandomNum( m_e, min_val, max_val );

        memset( m_y_out,      0, m_num_elements * sizeof(T) );
        memset( m_y_baseline, 0, m_num_elements * sizeof(T) );
    }

    virtual ~TestExecutorSAXPY() {
        delete[] m_x;
        delete[] m_y_org;
	delete[] m_y_out;
	delete[] m_y_baseline;
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
	auto t = dynamic_pointer_cast< TestCaseSAXPY<T> >( this->m_test_cases[ test_case ] );

	if ( test_case == 0 ) {
            memcpy( m_y_baseline, t->getY(), sizeof(T)*m_num_elements );
	}
        t->calculateRMS( m_y_baseline );
    }

    void prepareForRun ( const int test_case, const int num ) {

        memcpy( m_y_out, m_y_org, sizeof(T)*m_num_elements );
        auto t = dynamic_pointer_cast< TestCaseSAXPY<T> >( this->m_test_cases[ test_case ] );
	t->setX     ( m_x     );
        t->setY     ( m_y_out );
        t->setAlpha ( m_alpha );
    }
};

static const size_t NUM_METAL_THREADS_PER_GROUP = 1024;
static const size_t NUM_METAL_WARP_SIZE         =   32;
static const size_t NUM_TRIALS                  =  100;
#if TARGET_OS_OSX
size_t nums_elements[]{ 128, 512, 2*1024, 8*1024, 32*1024, 128*1024, 512*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024 };
#else
size_t nums_elements[]{ 128, 512, 2*1024, 8*1024, 32*1024, 128*1024, 512*1024, 4*1024*1024, 16*1024*1024 };
#endif
template<class T>
void testSuitePerType () {

    const int neon_num_lanes = (is_same<float, T>::value)?4:2;

    for( auto num_elements : nums_elements ) {

        TestExecutorSAXPY<T> e( cout, num_elements, NUM_TRIALS, false, -1.0, 1.0 );

        e.addTestCase( make_shared< TestCaseSAXPY_baseline               <T> > ( num_elements        ) );
        e.addTestCase( make_shared< TestCaseSAXPY_multithread_block      <T> > ( num_elements,     2 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_multithread_block      <T> > ( num_elements,     4 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_multithread_block      <T> > ( num_elements,     8 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_multithread_block      <T> > ( num_elements,    16 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_multithread_interleave <T> > ( num_elements,     2 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_multithread_interleave <T> > ( num_elements,     4 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_multithread_interleave <T> > ( num_elements,     8 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_multithread_interleave <T> > ( num_elements,    16 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_neon                   <T> > ( num_elements, 1     ) );
        e.addTestCase( make_shared< TestCaseSAXPY_neon                   <T> > ( num_elements, 2     ) );
        e.addTestCase( make_shared< TestCaseSAXPY_neon                   <T> > ( num_elements, 4     ) );
        if ( num_elements >= 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseSAXPY_neon                   <T> > ( num_elements, 8     ) );
        }
        if ( num_elements >= 8 * 2 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseSAXPY_neon_multithread_block <T> > ( num_elements, 8,  2 ) );
        }
        if ( num_elements >= 8 * 4 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseSAXPY_neon_multithread_block <T> > ( num_elements, 8,  4 ) );
        }
        if ( num_elements >= 8 * 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseSAXPY_neon_multithread_block <T> > ( num_elements, 8,  8 ) );
        }
        e.addTestCase( make_shared< TestCaseSAXPY_vDSP                   <T> > ( num_elements        ) );
        e.addTestCase( make_shared< TestCaseSAXPY_BLAS                   <T> > ( num_elements        ) );

        if constexpr ( is_same<float, T>::value ) {
            if ( num_elements <= NUM_METAL_THREADS_PER_GROUP ) {

                int adjusted_num_metal_threads_per_group = align_up( num_elements, NUM_METAL_WARP_SIZE );

                e.addTestCase( make_shared< TestCaseSAXPY_Metal<float> > ( num_elements, adjusted_num_metal_threads_per_group, 1 ) );
            }
            else {
                for (   size_t num_metal_groups_per_grid = num_elements / NUM_METAL_THREADS_PER_GROUP
                      ; num_metal_groups_per_grid > 0
                      ; num_metal_groups_per_grid /= NUM_METAL_THREADS_PER_GROUP ) {
               
                    e.addTestCase( make_shared< TestCaseSAXPY_Metal<float> > (
                        num_elements,
                        NUM_METAL_THREADS_PER_GROUP,
                        num_metal_groups_per_grid
                    ) );
                }
            }
        }
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

    testSuitePerType<float> ();

    cerr << "\n\nTesting for type double.\n\n";

    testSuitePerType<double> ();

    return 0;

}
