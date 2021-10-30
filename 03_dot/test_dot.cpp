#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <Accelerate/Accelerate.h>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"

#include "dot_metal_cpp.h"

template<class T>
class TestCaseDOT : public TestCaseWithTimeMeasurements {

  protected:
    const size_t        m_num_elements;
    const T*            m_x;
    const T*            m_y;
    T                   m_dot;

  public:
    TestCaseDOT( const size_t num_elements )
        :m_num_elements      ( num_elements )
    {
        if constexpr ( is_same<float, T>::value ) {

            setDataElementType( FLOAT );
        }
        else if constexpr ( is_same<double, T>::value ) {

            setDataElementType( DOUBLE );
        }
        else {
            assert(true);
        }

        setVector( num_elements );

        setVerificationType( DISTANCE );
    }

    virtual ~TestCaseDOT(){;}

    virtual void setX( const T* const x ){ m_x = x; }
    virtual void setY( const T* const y ){ m_y = y; }
    virtual T    getResult(){ return m_dot; }

    void calculateDistance( const T baseline ) {

         double dist = fabs( ( baseline - getResult() ) / baseline );
         this->setDist( dist );
    }

    virtual void run() = 0;
};


template<class T>
class TestCaseDOT_baseline : public TestCaseDOT<T> {

  public:
    TestCaseDOT_baseline( const size_t num_elements )
        :TestCaseDOT<T>( num_elements )
    {
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseDOT_baseline(){;}

    virtual void run(){

        this->m_dot = 0.0;

        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

            this->m_dot += this->m_x[i] * this->m_y[i];
        }
    }
};


template<class T>
class TestCaseDOT_neon : public TestCaseDOT<T> {

  protected:

    const size_t m_factor_loop_unrolling;

    void (TestCaseDOT_neon<T>::*m_calc_block)(T*, const int, const int);

    void calc_block_factor_1( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        if constexpr( is_same<float, T>::value ) {

            float32x4_t dot_quad1{0.0, 0.0, 0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one ; i+=4 ) {

                const float32x4_t x_quad1  = vld1q_f32( &this->m_x[i]  );
                const float32x4_t y_quad1  = vld1q_f32( &this->m_y[i]  );
                const float32x4_t xy_quad1 = vmulq_f32( x_quad1, y_quad1 );
                dot_quad1 = vaddq_f32( dot_quad1, xy_quad1 );
            } 
            *sum = dot_quad1[0] + dot_quad1[1] + dot_quad1[2] + dot_quad1[3];
        }
        else {
            float64x2_t dot_pair1{0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one ; i+=2 ) {

                const float64x2_t x_pair1  = vld1q_f64( &this->m_x[i]  );
                const float64x2_t y_pair1  = vld1q_f64( &this->m_y[i]  );
                const float64x2_t xy_pair1 = vmulq_f64( x_pair1, y_pair1 );
                dot_pair1 = vaddq_f64( dot_pair1, xy_pair1 );
            } 
            *sum = dot_pair1[0] + dot_pair1[1];
        }
    }

    void calc_block_factor_2( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        if constexpr( is_same<float, T>::value ) {

            float32x4_t dot_quad1{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad2{0.0, 0.0, 0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one ; i+=8 ) {

                const float32x4_t x_quad1  = vld1q_f32( &this->m_x[i]  );
                const float32x4_t x_quad2  = vld1q_f32( &this->m_x[i+4]  );
                const float32x4_t y_quad1  = vld1q_f32( &this->m_y[i]  );
                const float32x4_t y_quad2  = vld1q_f32( &this->m_y[i+4]  );
                const float32x4_t xy_quad1 = vmulq_f32( x_quad1, y_quad1 );
                const float32x4_t xy_quad2 = vmulq_f32( x_quad2, y_quad2 );
                dot_quad1 = vaddq_f32( dot_quad1, xy_quad1 );
                dot_quad2 = vaddq_f32( dot_quad2, xy_quad2 );
            } 
            *sum =   dot_quad1[0] + dot_quad1[1] + dot_quad1[2] + dot_quad1[3]
                   + dot_quad2[0] + dot_quad2[1] + dot_quad2[2] + dot_quad2[3];
        }
        else {
            float64x2_t dot_pair1{0.0, 0.0};
            float64x2_t dot_pair2{0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=4 ) {

                const float64x2_t x_pair1  = vld1q_f64( &this->m_x[i]  );
                const float64x2_t x_pair2  = vld1q_f64( &this->m_x[i+2]  );
                const float64x2_t y_pair1  = vld1q_f64( &this->m_y[i]  );
                const float64x2_t y_pair2  = vld1q_f64( &this->m_y[i+2]  );
                const float64x2_t xy_pair1 = vmulq_f64( x_pair1, y_pair1 );
                const float64x2_t xy_pair2 = vmulq_f64( x_pair2, y_pair2 );
                dot_pair1 = vaddq_f64( dot_pair1, xy_pair1 );
                dot_pair2 = vaddq_f64( dot_pair2, xy_pair2 );
            } 
            *sum = dot_pair1[0] + dot_pair1[1] +  dot_pair2[0] + dot_pair2[1];
        }
    }

    void calc_block_factor_4( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        if constexpr( is_same<float, T>::value ) {

            float32x4_t dot_quad1{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad2{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad3{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad4{0.0, 0.0, 0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=16 ) {

                const float32x4_t x_quad1  = vld1q_f32( &this->m_x[i]  );
                const float32x4_t x_quad2  = vld1q_f32( &this->m_x[i+4]  );
                const float32x4_t x_quad3  = vld1q_f32( &this->m_x[i+8]  );
                const float32x4_t x_quad4  = vld1q_f32( &this->m_x[i+12]  );
                const float32x4_t y_quad1  = vld1q_f32( &this->m_y[i]  );
                const float32x4_t y_quad2  = vld1q_f32( &this->m_y[i+4]  );
                const float32x4_t y_quad3  = vld1q_f32( &this->m_y[i+8]  );
                const float32x4_t y_quad4  = vld1q_f32( &this->m_y[i+12]  );
                const float32x4_t xy_quad1 = vmulq_f32( x_quad1, y_quad1 );
                const float32x4_t xy_quad2 = vmulq_f32( x_quad2, y_quad2 );
                const float32x4_t xy_quad3 = vmulq_f32( x_quad3, y_quad3 );
                const float32x4_t xy_quad4 = vmulq_f32( x_quad4, y_quad4 );
                dot_quad1 = vaddq_f32( dot_quad1, xy_quad1 );
                dot_quad2 = vaddq_f32( dot_quad2, xy_quad2 );
                dot_quad3 = vaddq_f32( dot_quad3, xy_quad3 );
                dot_quad4 = vaddq_f32( dot_quad4, xy_quad4 );
            } 
            *sum =   dot_quad1[0] + dot_quad1[1] + dot_quad1[2] + dot_quad1[3]
                   + dot_quad2[0] + dot_quad2[1] + dot_quad2[2] + dot_quad2[3]
                   + dot_quad3[0] + dot_quad3[1] + dot_quad3[2] + dot_quad3[3]
                   + dot_quad4[0] + dot_quad4[1] + dot_quad4[2] + dot_quad4[3];
        }
        else {
            float64x2_t dot_pair1{0.0, 0.0};
            float64x2_t dot_pair2{0.0, 0.0};
            float64x2_t dot_pair3{0.0, 0.0};
            float64x2_t dot_pair4{0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=8 ) {

                const float64x2_t x_pair1  = vld1q_f64( &this->m_x[i]  );
                const float64x2_t x_pair2  = vld1q_f64( &this->m_x[i+2]  );
                const float64x2_t x_pair3  = vld1q_f64( &this->m_x[i+4]  );
                const float64x2_t x_pair4  = vld1q_f64( &this->m_x[i+6]  );
                const float64x2_t y_pair1  = vld1q_f64( &this->m_y[i]  );
                const float64x2_t y_pair2  = vld1q_f64( &this->m_y[i+2]  );
                const float64x2_t y_pair3  = vld1q_f64( &this->m_y[i+4]  );
                const float64x2_t y_pair4  = vld1q_f64( &this->m_y[i+6]  );
                const float64x2_t xy_pair1 = vmulq_f64( x_pair1, y_pair1 );
                const float64x2_t xy_pair2 = vmulq_f64( x_pair2, y_pair2 );
                const float64x2_t xy_pair3 = vmulq_f64( x_pair3, y_pair3 );
                const float64x2_t xy_pair4 = vmulq_f64( x_pair4, y_pair4 );
                dot_pair1 = vaddq_f64( dot_pair1, xy_pair1 );
                dot_pair2 = vaddq_f64( dot_pair2, xy_pair2 );
                dot_pair3 = vaddq_f64( dot_pair3, xy_pair3 );
                dot_pair4 = vaddq_f64( dot_pair4, xy_pair4 );
            } 
            *sum =   dot_pair1[0] + dot_pair1[1] +  dot_pair2[0] + dot_pair2[1]
                   + dot_pair3[0] + dot_pair3[1] +  dot_pair4[0] + dot_pair4[1];
        }
    }

    void calc_block_factor_8( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        if constexpr( is_same<float, T>::value ) {

            float32x4_t dot_quad1{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad2{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad3{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad4{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad5{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad6{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad7{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad8{0.0, 0.0, 0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=32 ) {

                const float32x4_t x_quad1  = vld1q_f32( &this->m_x[i   ]  );
                const float32x4_t x_quad2  = vld1q_f32( &this->m_x[i+ 4]  );
                const float32x4_t x_quad3  = vld1q_f32( &this->m_x[i+ 8]  );
                const float32x4_t x_quad4  = vld1q_f32( &this->m_x[i+12]  );
                const float32x4_t x_quad5  = vld1q_f32( &this->m_x[i+16   ]  );
                const float32x4_t x_quad6  = vld1q_f32( &this->m_x[i+20]  );
                const float32x4_t x_quad7  = vld1q_f32( &this->m_x[i+24]  );
                const float32x4_t x_quad8  = vld1q_f32( &this->m_x[i+28]  );
                const float32x4_t y_quad1  = vld1q_f32( &this->m_y[i   ]  );
                const float32x4_t y_quad2  = vld1q_f32( &this->m_y[i+ 4]  );
                const float32x4_t y_quad3  = vld1q_f32( &this->m_y[i+ 8]  );
                const float32x4_t y_quad4  = vld1q_f32( &this->m_y[i+12]  );
                const float32x4_t y_quad5  = vld1q_f32( &this->m_y[i+16]  );
                const float32x4_t y_quad6  = vld1q_f32( &this->m_y[i+20]  );
                const float32x4_t y_quad7  = vld1q_f32( &this->m_y[i+24]  );
                const float32x4_t y_quad8  = vld1q_f32( &this->m_y[i+28]  );
                const float32x4_t xy_quad1 = vmulq_f32( x_quad1, y_quad1 );
                const float32x4_t xy_quad2 = vmulq_f32( x_quad2, y_quad2 );
                const float32x4_t xy_quad3 = vmulq_f32( x_quad3, y_quad3 );
                const float32x4_t xy_quad4 = vmulq_f32( x_quad4, y_quad4 );
                const float32x4_t xy_quad5 = vmulq_f32( x_quad5, y_quad5 );
                const float32x4_t xy_quad6 = vmulq_f32( x_quad6, y_quad6 );
                const float32x4_t xy_quad7 = vmulq_f32( x_quad7, y_quad7 );
                const float32x4_t xy_quad8 = vmulq_f32( x_quad8, y_quad8 );
                dot_quad1 = vaddq_f32( dot_quad1, xy_quad1 );
                dot_quad2 = vaddq_f32( dot_quad2, xy_quad2 );
                dot_quad3 = vaddq_f32( dot_quad3, xy_quad3 );
                dot_quad4 = vaddq_f32( dot_quad4, xy_quad4 );
                dot_quad5 = vaddq_f32( dot_quad5, xy_quad5 );
                dot_quad6 = vaddq_f32( dot_quad6, xy_quad6 );
                dot_quad7 = vaddq_f32( dot_quad7, xy_quad7 );
                dot_quad8 = vaddq_f32( dot_quad8, xy_quad8 );
            } 
            *sum =   dot_quad1[0] + dot_quad1[1] + dot_quad1[2] + dot_quad1[3]
                   + dot_quad2[0] + dot_quad2[1] + dot_quad2[2] + dot_quad2[3]
                   + dot_quad3[0] + dot_quad3[1] + dot_quad3[2] + dot_quad3[3]
                   + dot_quad4[0] + dot_quad4[1] + dot_quad4[2] + dot_quad4[3]
                   + dot_quad5[0] + dot_quad5[1] + dot_quad5[2] + dot_quad5[3]
                   + dot_quad6[0] + dot_quad6[1] + dot_quad6[2] + dot_quad6[3]
                   + dot_quad7[0] + dot_quad7[1] + dot_quad7[2] + dot_quad7[3]
                   + dot_quad8[0] + dot_quad8[1] + dot_quad8[2] + dot_quad8[3];
        }
        else {
            float64x2_t dot_pair1{0.0, 0.0};
            float64x2_t dot_pair2{0.0, 0.0};
            float64x2_t dot_pair3{0.0, 0.0};
            float64x2_t dot_pair4{0.0, 0.0};
            float64x2_t dot_pair5{0.0, 0.0};
            float64x2_t dot_pair6{0.0, 0.0};
            float64x2_t dot_pair7{0.0, 0.0};
            float64x2_t dot_pair8{0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=16 ) {

                const float64x2_t x_pair1  = vld1q_f64( &this->m_x[i   ]  );
                const float64x2_t x_pair2  = vld1q_f64( &this->m_x[i+ 2]  );
                const float64x2_t x_pair3  = vld1q_f64( &this->m_x[i+ 4]  );
                const float64x2_t x_pair4  = vld1q_f64( &this->m_x[i+ 6]  );
                const float64x2_t x_pair5  = vld1q_f64( &this->m_x[i+ 8]  );
                const float64x2_t x_pair6  = vld1q_f64( &this->m_x[i+10]  );
                const float64x2_t x_pair7  = vld1q_f64( &this->m_x[i+12]  );
                const float64x2_t x_pair8  = vld1q_f64( &this->m_x[i+14]  );
                const float64x2_t y_pair1  = vld1q_f64( &this->m_y[i   ]  );
                const float64x2_t y_pair2  = vld1q_f64( &this->m_y[i+ 2]  );
                const float64x2_t y_pair3  = vld1q_f64( &this->m_y[i+ 4]  );
                const float64x2_t y_pair4  = vld1q_f64( &this->m_y[i+ 6]  );
                const float64x2_t y_pair5  = vld1q_f64( &this->m_y[i+ 8]  );
                const float64x2_t y_pair6  = vld1q_f64( &this->m_y[i+10]  );
                const float64x2_t y_pair7  = vld1q_f64( &this->m_y[i+12]  );
                const float64x2_t y_pair8  = vld1q_f64( &this->m_y[i+14]  );
                const float64x2_t xy_pair1 = vmulq_f64( x_pair1, y_pair1 );
                const float64x2_t xy_pair2 = vmulq_f64( x_pair2, y_pair2 );
                const float64x2_t xy_pair3 = vmulq_f64( x_pair3, y_pair3 );
                const float64x2_t xy_pair4 = vmulq_f64( x_pair4, y_pair4 );
                const float64x2_t xy_pair5 = vmulq_f64( x_pair5, y_pair5 );
                const float64x2_t xy_pair6 = vmulq_f64( x_pair6, y_pair6 );
                const float64x2_t xy_pair7 = vmulq_f64( x_pair7, y_pair7 );
                const float64x2_t xy_pair8 = vmulq_f64( x_pair8, y_pair8 );
                dot_pair1 = vaddq_f64( dot_pair1, xy_pair1 );
                dot_pair2 = vaddq_f64( dot_pair2, xy_pair2 );
                dot_pair3 = vaddq_f64( dot_pair3, xy_pair3 );
                dot_pair4 = vaddq_f64( dot_pair4, xy_pair4 );
                dot_pair5 = vaddq_f64( dot_pair5, xy_pair5 );
                dot_pair6 = vaddq_f64( dot_pair6, xy_pair6 );
                dot_pair7 = vaddq_f64( dot_pair7, xy_pair7 );
                dot_pair8 = vaddq_f64( dot_pair8, xy_pair8 );
            } 
            *sum =   dot_pair1[0] + dot_pair1[1] +  dot_pair2[0] + dot_pair2[1]
                   + dot_pair3[0] + dot_pair3[1] +  dot_pair4[0] + dot_pair4[1]
                   + dot_pair5[0] + dot_pair5[1] +  dot_pair6[0] + dot_pair6[1]
                   + dot_pair7[0] + dot_pair7[1] +  dot_pair8[0] + dot_pair8[1];
        }
    }

  public:
    TestCaseDOT_neon( const size_t num_elements, const size_t factor_loop_unrolling )
        :TestCaseDOT<T>( num_elements )
        ,m_factor_loop_unrolling( factor_loop_unrolling )
    {
        this->setNEON( 1, factor_loop_unrolling );
        if (factor_loop_unrolling == 1) {
            m_calc_block = &TestCaseDOT_neon::calc_block_factor_1;
        }
        else if (factor_loop_unrolling == 2) {
            m_calc_block = &TestCaseDOT_neon::calc_block_factor_2;
        }
        else if (factor_loop_unrolling == 4) {
            m_calc_block = &TestCaseDOT_neon::calc_block_factor_4;
        }
        else if (factor_loop_unrolling == 8) {
            m_calc_block = &TestCaseDOT_neon::calc_block_factor_8;
        }
        else {
            assert(true);
        }
    }

    virtual ~TestCaseDOT_neon(){;}

    virtual inline void calc_block( T* sum, const int elem_begin, const int elem_end_past_one ) {
        (this->*m_calc_block)( sum, elem_begin, elem_end_past_one );
    }

    void run() {
        calc_block( &(this->m_dot), 0, this->m_num_elements );
    }
};


template<class T>
class TestCaseDOT_neon_multithread_block : public TestCaseDOT_neon<T> {

  protected:
    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    const int                   m_num_threads;
    vector<thread>              m_threads;
    T*                          m_partial_sums;

  public:
    TestCaseDOT_neon_multithread_block( const size_t num_elements, const size_t factor_loop_unrolling, const int num_threads )
        :TestCaseDOT_neon<T>( num_elements, factor_loop_unrolling )
        ,m_fan_out     ( num_threads )
        ,m_fan_in      ( num_threads )
        ,m_num_threads ( num_threads )
        ,m_partial_sums( new T[num_threads] )
    {
        this->setNEON( num_threads, factor_loop_unrolling );

        const size_t num_elems_per_thread = this->m_num_elements / m_num_threads;

        auto thread_lambda = [ this, num_elems_per_thread ]( const size_t thread_index ) {

            const size_t elem_begin        = thread_index * num_elems_per_thread;
            const size_t elem_end_past_one = elem_begin + num_elems_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                this->calc_block( &(m_partial_sums[thread_index]), elem_begin, elem_end_past_one );

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

    virtual ~TestCaseDOT_neon_multithread_block() {

        m_fan_out.terminate();
        m_fan_in.terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    void run(){
        memset( m_partial_sums, 0, sizeof(T) * m_num_threads );

        m_fan_out.notify();

        m_fan_in.wait();

        this->m_dot = 0.0;
        for ( size_t i = 0; i < m_num_threads; i++ ) {

            this->m_dot += m_partial_sums[i];
        }
    }
};


template<class T>
class TestCaseDOT_vDSP : public TestCaseDOT<T> {
  public:
    TestCaseDOT_vDSP( const size_t num_elements )
        :TestCaseDOT<T>( num_elements )
    {
        static_assert( is_same<float, T>::value || is_same<double, T>::value );
        this->setImplementationType(VDSP);
    }
    virtual ~TestCaseDOT_vDSP(){;}

    void run();
};

template<>
void TestCaseDOT_vDSP<float>::run()
{
    vDSP_dotpr ( this->m_x, 1, this->m_y, 1, &(this->m_dot), this->m_num_elements );
}

template<>
void TestCaseDOT_vDSP<double>::run()
{
    vDSP_dotprD ( this->m_x, 1, this->m_y, 1, &(this->m_dot), this->m_num_elements );
}


template<class T>
class TestCaseDOT_BLAS : public TestCaseDOT<T> {
  public:
    TestCaseDOT_BLAS( const size_t num_elements )
        :TestCaseDOT<T>( num_elements )
    {
        static_assert( is_same<float, T>::value || is_same<double, T>::value );
        this->setImplementationType(BLAS);
    }
    virtual ~TestCaseDOT_BLAS(){;}

    void run();
};

template<>
void TestCaseDOT_BLAS<float>::run()
{
    this->m_dot = cblas_sdot(this->m_num_elements, this->m_x, 1, this->m_y, 1);
}

template<>
void TestCaseDOT_BLAS<double>::run()
{
    this->m_dot = cblas_ddot(this->m_num_elements, this->m_x, 1, this->m_y, 1);
}


template<class T>
class TestCaseDOT_Metal : public TestCaseDOT<T> {

  private:
    DotMetalCpp m_metal;

  public:

    TestCaseDOT_Metal( const size_t num_elements , const size_t num_threads_per_group, const size_t num_groups_per_grid, const int reduction_type )
        :TestCaseDOT<T>( num_elements )
        ,m_metal( num_elements, num_threads_per_group, num_groups_per_grid, reduction_type )
    {
        static_assert( std::is_same<float, T>::value );

        switch( reduction_type ) {

          case 1:
            this->setMetal( TWO_PASS_DEVICE_MEMORY, num_groups_per_grid, num_threads_per_group );
            break;

          case 2:
            this->setMetal( TWO_PASS_SHARED_MEMORY, num_groups_per_grid, num_threads_per_group );
            break;

          case 3:
            this->setMetal( TWO_PASS_SIMD_SHUFFLE, num_groups_per_grid, num_threads_per_group );
            break;

          case 4:
            this->setMetal( TWO_PASS_SIMD_SUM, num_groups_per_grid, num_threads_per_group );
            break;

          case 5:
            this->setMetal( ONE_PASS_ATOMIC_SIMD_SHUFFLE, num_groups_per_grid, num_threads_per_group );
            break;

          case 6:
            this->setMetal( ONE_PASS_ATOMIC_SIMD_SUM, num_groups_per_grid, num_threads_per_group );
            break;

          case 7:
          default:
            this->setMetal( ONE_PASS_THREAD_COUNTER, num_groups_per_grid, num_threads_per_group );

        }
    }

    virtual ~TestCaseDOT_Metal(){;}

    void setX( const T* const x ){
        memcpy( m_metal.getRawPointerX(), x, this->m_num_elements * sizeof(T) );
    }

    void setY( const T* const y ){
        memcpy( m_metal.getRawPointerY(), y, this->m_num_elements * sizeof(T) );
    }

    void run() {
        m_metal.performComputation();
        this->m_dot = m_metal.getDotXY();
    }
};


template <class T>
class TestExecutorDOT : public TestExecutor {

  protected:

    const int             m_num_elements;
    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_x;
    T*                    m_y;
    T                     m_dot_baseline;

  public:

    TestExecutorDOT( ostream& os, const int num_elements, const int num_trials, const bool repeatable, const T min_val, const T max_val )
        :TestExecutor  ( os, num_trials )
        ,m_num_elements( num_elements )
        ,m_repeatable  ( repeatable )
        ,m_e( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_x           ( new T[num_elements] )
        ,m_y           ( new T[num_elements] )
        ,m_dot_baseline( 0.0)
    {
        fillArrayWithRandomValues( m_e, m_x, m_num_elements, min_val, max_val );
        fillArrayWithRandomValues( m_e, m_y, m_num_elements, min_val, max_val );
    }

    virtual ~TestExecutorDOT() {
        delete[] m_x;
        delete[] m_y;
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCaseDOT<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            m_dot_baseline = t->getResult();
        }
        t->calculateDistance( m_dot_baseline );
    }

    void prepareForRun ( const int test_case, const int num ) {

        auto t = dynamic_pointer_cast< TestCaseDOT<T> >( this->m_test_cases[ test_case ] );
        t->setX     ( m_x );
        t->setY     ( m_y );
    }
};


static const size_t NUM_METAL_THREADS_PER_GROUP = 1024;
static const size_t NUM_METAL_WARP_SIZE         =   32;
static const size_t NUM_TRIALS                  =  100;

size_t nums_elements[]{ 32, 128, 512, 2* 1024, 8*1024, 32*1024, 128*1024, 512*1024, 2*1024*1024, 8*1024*1024, 32*1024*1024, 128*1024*1024 };

template<class T>
void testSuitePerType () {

    const int neon_num_lanes = (is_same<float, T>::value)?4:2;

    for( auto num_elements : nums_elements ) {

        TestExecutorDOT<T> e( cout, num_elements, NUM_TRIALS, false, -1.0, 1.0 );

        e.addTestCase( make_shared< TestCaseDOT_baseline <T> > ( num_elements ) );

        e.addTestCase( make_shared< TestCaseDOT_neon <T> > ( num_elements,  1 ) );
        e.addTestCase( make_shared< TestCaseDOT_neon <T> > ( num_elements,  2 ) );
        e.addTestCase( make_shared< TestCaseDOT_neon <T> > ( num_elements,  4 ) );
        if ( num_elements >= 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDOT_neon <T> > ( num_elements,  8 ) );
        }

        if ( num_elements >= 2 * 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDOT_neon_multithread_block <T> > ( num_elements, 8, 2 ) );
        }
        if ( num_elements >= 4 * 8 * neon_num_lanes ) {        
            e.addTestCase( make_shared< TestCaseDOT_neon_multithread_block <T> > ( num_elements, 8, 4 ) );
        }
        if ( num_elements >= 8 * 8 * neon_num_lanes ) {        
            e.addTestCase( make_shared< TestCaseDOT_neon_multithread_block <T> > ( num_elements, 8, 8 ) );
        }
        e.addTestCase( make_shared< TestCaseDOT_vDSP              <T     > > ( num_elements ) );
        e.addTestCase( make_shared< TestCaseDOT_BLAS              <T     > > ( num_elements ) );

        if constexpr ( is_same<float, T>::value ) {
            if ( num_elements <= NUM_METAL_THREADS_PER_GROUP ) {

                int adjusted_num_metal_threads_per_group = align_up( num_elements, NUM_METAL_WARP_SIZE );

                e.addTestCase( make_shared< TestCaseDOT_Metal<float> > ( num_elements, adjusted_num_metal_threads_per_group, 1, 1 ) );
                e.addTestCase( make_shared< TestCaseDOT_Metal<float> > ( num_elements, adjusted_num_metal_threads_per_group, 1, 2 ) );
                e.addTestCase( make_shared< TestCaseDOT_Metal<float> > ( num_elements, adjusted_num_metal_threads_per_group, 1, 3 ) );
                e.addTestCase( make_shared< TestCaseDOT_Metal<float> > ( num_elements, adjusted_num_metal_threads_per_group, 1, 4 ) );
                e.addTestCase( make_shared< TestCaseDOT_Metal<float> > ( num_elements, adjusted_num_metal_threads_per_group, 1, 5 ) );
                e.addTestCase( make_shared< TestCaseDOT_Metal<float> > ( num_elements, adjusted_num_metal_threads_per_group, 1, 6 ) );
                e.addTestCase( make_shared< TestCaseDOT_Metal<float> > ( num_elements, adjusted_num_metal_threads_per_group, 1, 7 ) );
            }
            else {
                for (   size_t num_metal_groups_per_grid = num_elements / NUM_METAL_THREADS_PER_GROUP
                      ; num_metal_groups_per_grid > 0
                      ; num_metal_groups_per_grid /= NUM_METAL_THREADS_PER_GROUP ) {
               
                    e.addTestCase( make_shared< TestCaseDOT_Metal<float> > (
                        num_elements, NUM_METAL_THREADS_PER_GROUP, num_metal_groups_per_grid, 1 ) );
                    e.addTestCase( make_shared< TestCaseDOT_Metal<float> > (
                        num_elements, NUM_METAL_THREADS_PER_GROUP, num_metal_groups_per_grid, 2 ) );
                    e.addTestCase( make_shared< TestCaseDOT_Metal<float> > (
                        num_elements, NUM_METAL_THREADS_PER_GROUP, num_metal_groups_per_grid, 3 ) );
                    e.addTestCase( make_shared< TestCaseDOT_Metal<float> > (
                        num_elements, NUM_METAL_THREADS_PER_GROUP, num_metal_groups_per_grid, 4 ) );
                    e.addTestCase( make_shared< TestCaseDOT_Metal<float> > (
                        num_elements, NUM_METAL_THREADS_PER_GROUP, num_metal_groups_per_grid, 5 ) );
                    e.addTestCase( make_shared< TestCaseDOT_Metal<float> > (
                        num_elements, NUM_METAL_THREADS_PER_GROUP, num_metal_groups_per_grid, 6 ) );
                    e.addTestCase( make_shared< TestCaseDOT_Metal<float> > (
                        num_elements, NUM_METAL_THREADS_PER_GROUP, num_metal_groups_per_grid, 7 ) );
                }
            }
        }
        e.execute();
    }
}


int main(int argc, char* argv[]) {

    TestCaseWithTimeMeasurements::printHeader( cout );

    cerr << "\n\nTesting for type float.\n\n";

    testSuitePerType<float> ();

    cerr << "\n\nTesting for type double.\n\n";

    testSuitePerType<double> ();

    return 0;

}
