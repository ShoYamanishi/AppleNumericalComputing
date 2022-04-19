#include <iostream>
#include <memory>
#include "memcpy_metal_cpp.h"
#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"
template<class T>
class TestCaseMemcpy : public TestCaseWithTimeMeasurements {

  protected:
    const T*  m_in;
          T*  m_out;
    const int m_num_elements;

  public:
    TestCaseMemcpy( const size_t num_elements )
        :m_num_elements( num_elements )
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
    }

    virtual ~TestCaseMemcpy(){;}

    virtual void compareTruth( const T* truth ) {

        for (size_t i = 0; i < m_num_elements; i++) {

            if ( m_out[i] != truth[i] ) {
                setTrueFalse( false ) ;
            }
        }
        setTrueFalse( true) ;
    }

    virtual void setIn ( const T* const in  ){ m_in  = in;  }
    virtual void setOut(       T* const out ){ m_out = out; }
    virtual T*   getOut(){ return m_out; }

    virtual void run() = 0;
};


template<class T>
class TestCaseMemcpy_baseline : public TestCaseMemcpy<T> {

  public:

    TestCaseMemcpy_baseline( const size_t num_elements )
        :TestCaseMemcpy<T>( num_elements )
    {
        this->setCPPBlock( 1, 1 );
    }
    virtual ~TestCaseMemcpy_baseline(){;}

    void run(){
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
//            if (i % 1 == 0) {
//                __builtin_prefetch (&(this->m_out[i+1]), 1, 0);
//                __builtin_prefetch (&(this->m_in [i+1]), 0, 0);
//            }
            this->m_out[i] = this->m_in[i];
        }
    }
};


template<class T>
class TestCaseMemcpy_loop_unrolled : public TestCaseMemcpy<T> {

  protected:

    const T*  m_in_aligned;
    T*        m_out_aligned;
    const int m_factor_loop_unrolling;

  public:


    TestCaseMemcpy_loop_unrolled( const size_t num_elements, const int factor_loop_unrolling )
        :TestCaseMemcpy<T>( num_elements )
        ,m_in_aligned ( nullptr )
        ,m_out_aligned( nullptr )
        ,m_factor_loop_unrolling( factor_loop_unrolling )
    {
        assert( factor_loop_unrolling == 1 || factor_loop_unrolling == 2 || factor_loop_unrolling == 4 || factor_loop_unrolling == 8 );
        this->setCPPBlock( 1, factor_loop_unrolling );
    }
    virtual ~TestCaseMemcpy_loop_unrolled(){;}

    virtual void setIn ( const T* const __attribute__((aligned(64))) in  ){ m_in_aligned  = in;  }
    virtual void setOut(       T* const __attribute__((aligned(64))) out ){ m_out_aligned = out; }
    virtual T*   getOut(){ return m_out_aligned; }

    virtual void compareTruth( const T* truth ) {

        for (size_t i = 0; i < this->m_num_elements; i++) {

            if ( m_out_aligned[i] != truth[i] ) {
                this->setTrueFalse( false ) ;
            }
        }
        this->setTrueFalse( true) ;
    }


    void run() {
        process_block( 0, this->m_num_elements );
    }

    inline void process_block(const int elem_begin, const int elem_end_past_one ) {

        switch ( m_factor_loop_unrolling ) {

          case 1:
            // #pragma unroll 1
            for ( size_t i = elem_begin; i < elem_end_past_one; i++ ) {

                const T* const __attribute__((aligned(4))) in  = &(this->m_in_aligned [i]);
                T* const       __attribute__((aligned(4))) out = &(this->m_out_aligned[i]);

                out[0] = in[0];
            }
            break;

          case 2:
            // #pragma unroll 2
            for ( size_t i = elem_begin; i < elem_end_past_one; i+=2 ) {

                const T* const __attribute__((aligned(8))) in  = &(this->m_in_aligned [i]);
                T* const       __attribute__((aligned(8))) out = &(this->m_out_aligned[i]);

                out[0] = in[0];
                out[1] = in[1];
            }
            break;

          case 4:
            // #pragma unroll 4
            for ( size_t i = elem_begin; i < elem_end_past_one; i+=4 ) {

                const T* const __attribute__((aligned(16))) in  = &(this->m_in_aligned [i]);
                T* const       __attribute__((aligned(16))) out = &(this->m_out_aligned[i]);

                out[0] = in[0];
                out[1] = in[1];
                out[2] = in[2];
                out[3] = in[3];
            }
            break;

          case 8:
          default:
            // #pragma unroll 8
            for ( size_t i = elem_begin; i < elem_end_past_one; i+=8 ) {

                const T* const __attribute__((aligned(32))) in  = &(this->m_in_aligned [i]);
                T* const       __attribute__((aligned(32))) out = &(this->m_out_aligned[i]);

                out[0] = in[0];
                out[1] = in[1];
                out[2] = in[2];
                out[3] = in[3];
                out[4] = in[4];
                out[5] = in[5];
                out[6] = in[6];
                out[7] = in[7];
            }
        }
    }
};

template<class T>
class TestCaseMemcpy_multithread : public TestCaseMemcpy_loop_unrolled<T> {

  protected:
    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    const int                   m_num_threads;
    vector<thread>              m_threads;

  public:
    TestCaseMemcpy_multithread( const size_t num_elements, const int factor_loop_unrolling, const int num_threads )
        :TestCaseMemcpy_loop_unrolled<T>( num_elements, factor_loop_unrolling )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
        ,m_num_threads( num_threads )
    {
        this->setCPPBlock( num_threads, factor_loop_unrolling );

        const size_t num_elems_per_thread = this->m_num_elements / m_num_threads;

        auto thread_lambda = [this, num_elems_per_thread ]( const size_t thread_index ) {

            const size_t elem_begin        = thread_index * num_elems_per_thread;
            const size_t elem_end_past_one = elem_begin + num_elems_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                this->process_block( elem_begin, elem_end_past_one );

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

    virtual ~TestCaseMemcpy_multithread(){

        m_fan_out.terminate();
        m_fan_in.terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    void run() {

        m_fan_out.notify();

        m_fan_in.wait();
    }
};


template<class T>
class TestCaseMemcpy_memcpy : public TestCaseMemcpy<T> {

  public:
    TestCaseMemcpy_memcpy( const size_t num_elements ):TestCaseMemcpy<T>( num_elements ) {
        this->setMemcpy( 1 );
    }

    virtual ~TestCaseMemcpy_memcpy(){;}

    void run(){
        memcpy( this->m_out, this->m_in, sizeof(T) * this->m_num_elements );
    }
};


template<class T>
class TestCaseMemcpy_memcpy_multithread : public TestCaseMemcpy<T> {

  protected:
    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    const int                   m_num_threads;
    vector<thread>              m_threads;

  public:
    TestCaseMemcpy_memcpy_multithread( const size_t num_elements, const int num_threads )
        :TestCaseMemcpy<T>( num_elements )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
        ,m_num_threads( num_threads )
    {
        this->setMemcpy( num_threads );

        const size_t num_elems_per_thread = this->m_num_elements / m_num_threads;

        auto thread_lambda = [this, num_elems_per_thread ]( const size_t thread_index ) {

            const size_t elem_begin        = thread_index * num_elems_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                memcpy( &(this->m_out[elem_begin]), &(this->m_in[elem_begin]), sizeof(T) * num_elems_per_thread );

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

    virtual ~TestCaseMemcpy_memcpy_multithread(){

        m_fan_out.terminate();
        m_fan_in.terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    void run() {

        m_fan_out.notify();

        m_fan_in.wait();
    }
};


template<class T>
class TestCaseMemcpy_metal : public TestCaseMemcpy<T> {

  protected:
    MemcpyMetalCpp m_metal;

  public:
    TestCaseMemcpy_metal( const size_t num_elements, const bool use_managed_buffer )
        :TestCaseMemcpy<T>( num_elements )
        ,m_metal( num_elements * sizeof(T), use_managed_buffer )
    {;}

    virtual ~TestCaseMemcpy_metal(){;}

    void compareTruth( const T* truth ){
        this->m_out = (T*)m_metal.getRawPointerOut();
        TestCaseMemcpy<T>::compareTruth( truth );
    }

    void setIn( const T* const x ){
        memcpy( m_metal.getRawPointerIn(), x, this->m_num_elements*sizeof(T) );
    }

    void setOut( T* const y ){
        memcpy( m_metal.getRawPointerOut(), y, this->m_num_elements*sizeof(T) );
    }
};


template<class T>
class TestCaseMemcpy_metal_kernel : public TestCaseMemcpy_metal<T> {

  public:
    TestCaseMemcpy_metal_kernel( const size_t num_elements, const bool use_managed_buffer )
        :TestCaseMemcpy_metal<T>( num_elements, use_managed_buffer )
    {
        this->setMetal(
            use_managed_buffer ? DEFAULT_MANAGED : DEFAULT_SHARED,
            this->m_metal.numGroupsPerGrid(),
            this->m_metal.numThreadsPerGroup()
        );
    }

    virtual ~TestCaseMemcpy_metal_kernel(){;}

    void run() {
        this->m_metal.performComputationKernel();
    }
};


template<class T>
class TestCaseMemcpy_metal_blit : public TestCaseMemcpy_metal<T> {

  public:
    TestCaseMemcpy_metal_blit( const size_t num_elements, const bool use_managed_buffer )
        :TestCaseMemcpy_metal<T>( num_elements, use_managed_buffer )
    {
        this->setMetal( use_managed_buffer ? BLIT_MANAGED: BLIT_SHARED, 0, 0 );
    }

    virtual ~TestCaseMemcpy_metal_blit(){;}

    void run() {
        this->m_metal.performComputationBlit();
    }
};


template <class T>
class TestExecutorMemcpy : public TestExecutor {

  protected:

    const int             m_num_elements;
    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_in;
    T*                    m_out;
    T*                    m_out_baseline;

  public:

    TestExecutorMemcpy( ostream& os, const int num_elements, const int num_trials, const bool repeatable, const T min_val, const T max_val )
        :TestExecutor  ( os, num_trials )
        ,m_num_elements( num_elements )
        ,m_repeatable  ( repeatable )
        ,m_e( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_in          ( new T[num_elements] )
        ,m_out         ( new T[num_elements] )
        ,m_out_baseline( new T[num_elements] )
    {
        fillArrayWithRandomValues( m_e, m_in, m_num_elements, min_val, max_val );
        memset( m_out,          0, m_num_elements * sizeof(T) );
        memset( m_out_baseline, 0, m_num_elements * sizeof(T) );
    }

    virtual ~TestExecutorMemcpy() {
        delete[] m_in;
        delete[] m_out;
        delete[] m_out_baseline;
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast<TestCaseMemcpy<T>>( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            memcpy( m_out_baseline, t->getOut(), sizeof(T)*m_num_elements );
        }

        t->compareTruth( m_out_baseline );
    }

    void prepareForRun ( const int test_case, const int num ) {

        memset( m_out, 0, sizeof(T)*m_num_elements );
        auto t = dynamic_pointer_cast<TestCaseMemcpy<T>>( this->m_test_cases[ test_case ] );
        t->setIn ( m_in  );
        t->setOut( m_out );
    }
};


static const size_t NUM_TRIALS = 10;
#if TARGET_OS_OSX
size_t nums_elements[] = { 32, 64, 128, 256, 512, 1024,
                      2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 
                     1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024, 16*1024*1024, 32*1024*1024, 64*1024*1024, 128*1024*1024 };
#else
size_t nums_elements[] = { 32, 64, 128, 256, 512, 1024,
                      2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024,
                     1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024, 16*1024*1024, 32*1024*1024 };
#endif

#if TARGET_OS_OSX
int main( int argc, char* argv[] ) {
#else
int run_test() {
#endif

    TestCaseWithTimeMeasurements::printHeader( cout );

    for ( auto n : nums_elements ) {

        TestExecutorMemcpy<int> e( cout, n, NUM_TRIALS, false , INT_MIN, INT_MAX );

        e.addTestCase( make_shared< TestCaseMemcpy_baseline <int> > ( n ) );
        e.addTestCase( make_shared< TestCaseMemcpy_loop_unrolled <int> > ( n, 2 ) );
        e.addTestCase( make_shared< TestCaseMemcpy_loop_unrolled <int> > ( n, 4 ) );
        e.addTestCase( make_shared< TestCaseMemcpy_loop_unrolled <int> > ( n, 8 ) );

        if ( n >= sizeof(int) * 8 * 2) {
            e.addTestCase( make_shared< TestCaseMemcpy_multithread <int> > ( n, 4 , 2) );
        }
        if ( n >= sizeof(int) * 8 * 4) {
            e.addTestCase( make_shared< TestCaseMemcpy_multithread <int> > ( n, 4 , 4) );
        }
        if ( n >= sizeof(int) * 8 * 8) {
            e.addTestCase( make_shared< TestCaseMemcpy_multithread <int> > ( n, 4 , 8) );
        }
        e.addTestCase( make_shared< TestCaseMemcpy_memcpy <int> > ( n ) );
        e.addTestCase( make_shared< TestCaseMemcpy_memcpy_multithread <int> > ( n , 2 ) );
        e.addTestCase( make_shared< TestCaseMemcpy_memcpy_multithread <int> > ( n , 4 ) );
        e.addTestCase( make_shared< TestCaseMemcpy_memcpy_multithread <int> > ( n , 8 ) );
        e.addTestCase( make_shared< TestCaseMemcpy_metal_kernel <int> > ( n , false ) );
#if TARGET_OS_OSX
        e.addTestCase( make_shared< TestCaseMemcpy_metal_kernel <int> > ( n , true  ) );
#endif
        e.addTestCase( make_shared< TestCaseMemcpy_metal_blit   <int> > ( n , false ) );
#if TARGET_OS_OSX
        e.addTestCase( make_shared< TestCaseMemcpy_metal_blit   <int> > ( n , true  ) );
#endif
        e.execute();
    }
    return 0;
}
