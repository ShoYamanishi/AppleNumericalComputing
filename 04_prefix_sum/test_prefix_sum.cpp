#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>


#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"

#include "prefix_sum_metal_cpp.h"


template<class T>
class TestCasePrefixSum : public TestCaseWithTimeMeasurements {

  protected:
    const size_t        m_num_elements;
    const T*            m_in;
          T*            m_out;
    double              m_dist_from_baseline;

  public:

    TestCasePrefixSum( const size_t num_elements )
        :m_num_elements      ( num_elements )
        ,m_dist_from_baseline( 0.0 )
    {
        if constexpr ( is_same<int, T>::value ) {

            setDataElementType( INT );
        }
        else {
            assert(true);
        }

        setVector( num_elements );

        setVerificationType( DISTANCE );
    }

    virtual ~TestCasePrefixSum(){;}

    void calculateDistance( const T* baseline ) {
        double sum = 0.0;

        m_out = getOut();
        for ( size_t i = 0; i < m_num_elements; i++ ) {

            sum += fabs( (double)( m_out[i] - baseline[i] ) );
//            cerr << "baseline[" << i << "]:" << baseline[i] << "\t";
//            cerr << "m_out[" << i << "]:" << m_out[i] << "\n";
        }
        this->setDist( sum /(double)(m_num_elements));
    }


    virtual void setIn( const T* const in  ){ m_in  = in; }
    virtual void setOut(      T* const out ){ m_out = out; }
    virtual T* getOut(){ return m_out; }

    virtual void copyBackOut(){;}

    virtual void run() = 0;
};


template<class T>
class TestCasePrefixSum_baseline : public TestCasePrefixSum<T> {

  protected:
    const size_t m_factor_loop_unrolling;

  public:
    TestCasePrefixSum_baseline( const size_t num_elements, const int factor_loop_unrolling )
        :TestCasePrefixSum<T>( num_elements )
        ,m_factor_loop_unrolling( factor_loop_unrolling )
    {
        assert(    factor_loop_unrolling == 1
                || factor_loop_unrolling == 2
                || factor_loop_unrolling == 4
                || factor_loop_unrolling == 8 );

        this->setCPPBlock( 1, factor_loop_unrolling );
    }

    virtual ~TestCasePrefixSum_baseline(){;}

    void run() {

        if ( m_factor_loop_unrolling == 1 ) {

            T sum = 0;
            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                this->m_out[i] = this->m_in[i] + sum ;
                sum = this->m_out[i];
            }
        }
        else if ( m_factor_loop_unrolling == 2 ) {

            T sum = 0;
            for ( size_t i = 0; i < this->m_num_elements ; i+= 2 ) {

                this->m_out[i  ] = this->m_in[i] + sum ;
                this->m_out[i+1] = this->m_in[i] + this->m_in[i+1] + sum ;

                sum = this->m_out[i+1];
            }
        }
        else if ( m_factor_loop_unrolling == 4 ) {

            T sum = 0;
            for ( size_t i = 0; i < this->m_num_elements ; i+= 4 ) {

                this->m_out[i  ] = this->m_in[i] + sum ;
                this->m_out[i+1] = this->m_in[i] + this->m_in[i+1] + sum ;
                this->m_out[i+2] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + sum ;
                this->m_out[i+3] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3] + sum ;

                sum = this->m_out[i+3];
            }
        }
        else if ( m_factor_loop_unrolling == 8 ) {

            T sum = 0;
            for ( size_t i = 0; i < this->m_num_elements ; i+= 8 ) {

                this->m_out[i  ] = this->m_in[i] + sum ;
                this->m_out[i+1] = this->m_in[i] + this->m_in[i+1] + sum ;
                this->m_out[i+2] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + sum ;
                this->m_out[i+3] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3] + sum ;
                this->m_out[i+4] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3]
                                    + this->m_in[i+4] + sum ;

                this->m_out[i+5] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3]
                                    + this->m_in[i+4] + this->m_in[i+5] + sum ;

                this->m_out[i+6] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3]
                                    + this->m_in[i+4] + this->m_in[i+5] + this->m_in[i+6] + sum ;

                this->m_out[i+7] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3]
                                    + this->m_in[i+4] + this->m_in[i+5] + this->m_in[i+6] + this->m_in[i+7] + sum ;

                sum = this->m_out[i+7];
            }
        }
    }
};


template<class T>
class TestCasePrefixSum_stdcpp : public TestCasePrefixSum<T> {

  protected:
    vector<T> m_in_vector;
    vector<T> m_out_vector;

  public:
    TestCasePrefixSum_stdcpp( const size_t num_elements )
        :TestCasePrefixSum<T>( num_elements )
    {
        this->setImplementationType( CPP_STDLIB );
    }

    virtual ~TestCasePrefixSum_stdcpp(){;}

    virtual void run() {
        std::inclusive_scan( m_in_vector.begin(), m_in_vector.end(), m_out_vector.begin() );
    }

    virtual void setIn( const T* const in  )
    {
        TestCasePrefixSum<T>::setIn( in );

        m_in_vector.clear();
        for ( int i = 0; i < this->m_num_elements; i++ ) {
            m_in_vector.push_back( this->m_in[i] );
        }
    }

    virtual void setOut( T* const out  )
    {
        TestCasePrefixSum<T>::setOut( out );
        m_out_vector.clear();
        for ( int i = 0; i < this->m_num_elements; i++ ) {
            m_out_vector.push_back( 0.0 );
        }
    }

    virtual T* getOut()
    {
        for ( int i = 0; i < this->m_num_elements; i++ ) {
            this->m_out[i] = m_out_vector[i];
        }
        return TestCasePrefixSum<T>::getOut();
    }
};


template<class T>
class TestCasePrefixSum_Metal : public TestCasePrefixSum<T> {

  private:
    PrefixSumMetalCpp<T> m_metal;

  public:

    TestCasePrefixSum_Metal(
        const size_t num_elements,
        const int    algo_type,
        const size_t num_partial_sums,
        const uint   num_threads_per_threadgroup
    )
        :TestCasePrefixSum<T>( num_elements )
        ,m_metal( num_elements, algo_type, num_partial_sums, num_threads_per_threadgroup )
    {
        static_assert( is_same<int, T>::value );

        auto num_groups_per_grid   = m_metal.numGroupsPerGrid(1);
        auto num_threads_per_group = m_metal.numThreadsPerGroup(1);

        switch( algo_type ) {

          case 1:
            this->setMetal( SCAN_THEN_FAN, num_groups_per_grid, num_threads_per_group );
            break;

          case 2:
            this->setMetal( REDUCE_THEN_SCAN, num_groups_per_grid, num_threads_per_group );
            break;

          case 3:
          default:
            this->setMetal( MERRILL_AND_GRIMSHAW, num_partial_sums,  num_threads_per_threadgroup );
            break;
        }
    }

    virtual ~TestCasePrefixSum_Metal(){;}

    void setIn( const T* const a_in ){
        memcpy( m_metal.getRawPointerIn(),  a_in,  this->m_num_elements * sizeof(T) );
    }

    void setOut( T* const a_out ){
        memcpy( m_metal.getRawPointerOut(), a_out, this->m_num_elements * sizeof(T) );
    }

    virtual T* getOut() {
        return m_metal.getRawPointerOut();
    }

    void copyBackOut() {
        memcpy( this->m_out, m_metal.getRawPointerOut(), this->m_num_elements * sizeof(T) );
    }

    void run() {
        m_metal.performComputation();
    }
};


template <class T>
class TestExecutorPrefixSum : public TestExecutor {

  protected:

    const int             m_num_elements;
    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_in;
    T*                    m_out;
    T*                    m_out_baseline;

  public:

    TestExecutorPrefixSum( ostream& os, const int num_elements, const int num_trials, const bool repeatable, const T min_val, const T max_val )
        :TestExecutor  ( os, num_trials )
        ,m_num_elements( num_elements )
        ,m_repeatable  ( repeatable )
        ,m_e( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_in           ( new T[num_elements] )
        ,m_out          ( new T[num_elements] )
        ,m_out_baseline ( new T[num_elements] )
    {
        fillArrayWithRandomValues( m_e, m_in,  m_num_elements, min_val, max_val );
    }

    virtual ~TestExecutorPrefixSum() {
        delete[] m_in;
        delete[] m_out;
        delete[] m_out_baseline;
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCasePrefixSum <T> >( this->m_test_cases[ test_case ] );
        if ( test_case == 0 ) {
            memcpy( m_out_baseline , t->getOut(), sizeof(T)*m_num_elements );
        }
        t->calculateDistance( m_out_baseline ); 
    }

    void prepareForRun ( const int test_case, const int num ) {
        auto t = dynamic_pointer_cast< TestCasePrefixSum<T> >( this->m_test_cases[ test_case ] );
        memset( m_out, 0, sizeof(T) * this->m_num_elements );
        t->setIn     ( m_in  );
        t->setOut    ( m_out );
    }
};


static const size_t NUM_TRIALS = 100;

#if TARGET_OS_OSX

size_t nums_elements[]{ 32, 128, 512, 2*1024, 32*1024, 128*1024, 512*1024, 2*1024*1024, 32*1024*1024, 128*1024*1024 };
uint num_threads_per_threadgroup = 1024;

#else

size_t nums_elements[]{ 32, 128, 512, 2*1024, 32*1024, 128*1024, 512*1024, 2*1024*1024, 32*1024*1024 };
uint num_threads_per_threadgroup = 512;

#endif

template<class T>
void testSuitePerType () {

    for( auto num_elements : nums_elements ) {

        TestExecutorPrefixSum<T> e( cout, num_elements, NUM_TRIALS, false, 0, 10 );

        e.addTestCase( make_shared< TestCasePrefixSum_baseline <T> > ( num_elements, 1       ) );
        e.addTestCase( make_shared< TestCasePrefixSum_baseline <T> > ( num_elements, 2       ) );
        e.addTestCase( make_shared< TestCasePrefixSum_baseline <T> > ( num_elements, 4       ) );
        e.addTestCase( make_shared< TestCasePrefixSum_baseline <T> > ( num_elements, 8       ) );
        e.addTestCase( make_shared< TestCasePrefixSum_stdcpp   <T> > ( num_elements          ) );
        e.addTestCase( make_shared< TestCasePrefixSum_Metal    <T> > ( num_elements, 1,    0, num_threads_per_threadgroup ) );
        e.addTestCase( make_shared< TestCasePrefixSum_Metal    <T> > ( num_elements, 2,    0, num_threads_per_threadgroup ) );
        e.addTestCase( make_shared< TestCasePrefixSum_Metal    <T> > ( num_elements, 3,   64, num_threads_per_threadgroup ) );
        e.addTestCase( make_shared< TestCasePrefixSum_Metal    <T> > ( num_elements, 3, 1024, num_threads_per_threadgroup ) );

        e.execute();
    }
}

#if TARGET_OS_OSX
int main(int argc, char* argv[]) {
#else
int run_test() {
#endif
    cerr << "\n\nTesting for type int.\n\n";

    TestCaseWithTimeMeasurements::printHeader( cout );

    testSuitePerType<int> ();

    return 0;
}
