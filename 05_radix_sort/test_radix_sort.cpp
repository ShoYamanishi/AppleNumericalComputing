#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"

#include <boost/sort/spreadsort/spreadsort.hpp>
#include <boost/sort/sort.hpp>

#include "radix_sort_metal_cpp.h"

template<class T>
class TestCaseRadixSort : public TestCaseWithTimeMeasurements {

  protected:
    const size_t        m_num_elements;
    T*                  m_array;
    bool                m_calculation_correct;
  public:

    TestCaseRadixSort( const size_t num_elements )
        :m_num_elements       ( num_elements )
        ,m_calculation_correct( false )
    {
        if constexpr ( is_same<int, T>::value ) {

            setDataElementType( INT );
        }
        if constexpr ( is_same<float, T>::value ) {

            setDataElementType( FLOAT );
        }
        else {
            assert(true);
        }

        setVector( num_elements );

        setVerificationType( TRUE_FALSE );
    }

    virtual ~TestCaseRadixSort(){;}

    void checkResult( const T* truth )
    {
        T* out = getArray();

        setTrueFalse( true );       
        for ( size_t i = 0; i < m_num_elements; i++ ) {

//            cerr << "out[" << i << "]:" << out[i] << "\ttruth[" << i << "]:" << truth[i] << "\n";
            if ( out[i] != truth[i] ) {
                setTrueFalse( false );
            }
        }
    }


    virtual void setArray( T* const a ){ m_array = a; }
    virtual T* getArray(){ return m_array; }

    virtual void run() = 0;
};


template<class T>
class TestCaseRadixSort_baseline : public TestCaseRadixSort<T> {

  public:
    TestCaseRadixSort_baseline( const size_t num_elements )
        :TestCaseRadixSort<T>( num_elements )
    {
        this->setImplementationType( CPP_STDLIB );
    }

    virtual ~TestCaseRadixSort_baseline(){;}

    void run() {

        std::sort( this->m_array, this->m_array + this->m_num_elements );
    }
};


template<class T>
class TestCaseRadixSort_boost_spread_sort : public TestCaseRadixSort<T> {

  public:
    TestCaseRadixSort_boost_spread_sort( const size_t num_elements )
        :TestCaseRadixSort<T>( num_elements )
    {
        this->setImplementationType(BOOST_SPREAD_SORT);
    }

    virtual ~TestCaseRadixSort_boost_spread_sort(){;}

    void run(){

        if constexpr ( is_same<float, T>::value || is_same<double, T>::value ) {
            boost::sort::spreadsort::float_sort ( this->m_array, this->m_array + this->m_num_elements );
        }
        else {
            boost::sort::spreadsort::integer_sort( this->m_array, this->m_array + this->m_num_elements );
        }
    }
};


template<class T>
class TestCaseRadixSort_boost_sample_sort : public TestCaseRadixSort<T> {

  private:
    size_t m_num_threads;

  public:
    TestCaseRadixSort_boost_sample_sort( const size_t num_element, const size_t num_threads )
        :TestCaseRadixSort<T>( num_element )
        ,m_num_threads( num_threads )
    {
        this->setBoostSampleSort( num_threads );
    }

    virtual ~TestCaseRadixSort_boost_sample_sort(){;}

    void run() {

        boost::sort::block_indirect_sort( this->m_array, this->m_array + this->m_num_elements , m_num_threads );
    }
};


template<class T>
class TestCaseRadixSort_Metal : public TestCaseRadixSort<T> {

  private:
    RadixSortMetalCpp m_metal;

  public:
    TestCaseRadixSort_Metal(
        const size_t num_elements, 
        const bool   coalesced_write, 
        const bool   early_out, 
        const bool   in_one_commit,
        const size_t num_threads_per_threadgroup )
        :TestCaseRadixSort<T>( num_elements )
        ,m_metal( num_elements, is_same<float, T>::value, coalesced_write, early_out, in_one_commit, num_threads_per_threadgroup )
    {
        static_assert( std::is_same<int, T>::value || std::is_same<long, T>::value || std::is_same<float,T>::value );

        if ( early_out ) {
            this->setMetal( coalesced_write ? COALESCED_WRITE_EARLY_OUT : UNCOALESCED_WRITE_EARLY_OUT, 1, num_threads_per_threadgroup );
        }
        else if ( in_one_commit ) {
            this->setMetal( coalesced_write ? COALESCED_WRITE_IN_ONE_COMMIT : UNCOALESCED_WRITE_IN_ONE_COMMIT, 1, num_threads_per_threadgroup );
        }
        else {
            this->setMetal( coalesced_write ? COALESCED_WRITE : UNCOALESCED_WRITE, 1, num_threads_per_threadgroup );
        }
    }

    virtual ~TestCaseRadixSort_Metal(){;}

    void setArray( T* const x ) {
        m_metal.resetBufferFlag();
        memcpy( m_metal.getRawPointerIn(), x, this->m_num_elements*sizeof(T) );
    }

    T* getArray() {
        return (T*)m_metal.getRawPointerOut();
    }

    void run() {
        m_metal.performComputation();
    }
};


template <class T>
class TestExecutorRadixSort : public TestExecutor {

  protected:

    const int             m_num_elements;
    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_array_original;
    T*                    m_array_sorted_baseline;
    T*                    m_array_sorted;

  public:

    TestExecutorRadixSort( ostream& os, const int num_elements, const int num_trials, const bool repeatable, const T min_val, const T max_val )
        :TestExecutor  ( os, num_trials )
        ,m_num_elements( num_elements )
        ,m_repeatable  ( repeatable )
        ,m_e( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_array_original        ( new T[num_elements] )
        ,m_array_sorted_baseline ( new T[num_elements] )
        ,m_array_sorted          ( new T[num_elements] )
    {
        fillArrayWithRandomValues( m_e, m_array_original,  m_num_elements, min_val, max_val );
    }

    virtual ~TestExecutorRadixSort() {
        delete[] m_array_original;
        delete[] m_array_sorted_baseline;
        delete[] m_array_sorted;
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCaseRadixSort <T> >( this->m_test_cases[ test_case ] );
        if ( test_case == 0 ) {
            memcpy( m_array_sorted_baseline , t->getArray(), sizeof(T)*m_num_elements );
        }
        t->checkResult(  m_array_sorted_baseline );
    }

    void prepareForRun ( const int test_case, const int num ) {

        auto t = dynamic_pointer_cast< TestCaseRadixSort<T> >( this->m_test_cases[ test_case ] );

        memcpy( m_array_sorted, m_array_original, sizeof(T) * this->m_num_elements );

        t->setArray( m_array_sorted );
    }
};

static const size_t NUM_TRIALS = 10;

#if TARGET_OS_OSX

size_t nums_elements[]{ 32, 128, 512, 2*1024, 8*1024, 32*1024, 128*1024, 512*1024, 2*1024*1024, 8*1024*1024, 32*1024*1024, 128*1024*1024 };
uint num_threads_per_threadgroup = 1024;

#else
size_t nums_elements[]{ 32, 128, 512, 2*1024, 8*1024, 32*1024, 128*1024, 512*1024, 2*1024*1024, 8*1024*1024 };
uint num_threads_per_threadgroup = 1024;

#endif

template<class T>
void testSuitePerType () {

    for( auto num_elements : nums_elements ) {

        TestExecutorRadixSort<T> e( cout, num_elements, NUM_TRIALS, false, INT_MIN, INT_MAX );

        e.addTestCase( make_shared< TestCaseRadixSort_baseline <T> > ( num_elements ) );
        e.addTestCase( make_shared< TestCaseRadixSort_boost_spread_sort  <T> > ( num_elements         ) );
        e.addTestCase( make_shared< TestCaseRadixSort_boost_sample_sort  <T> > ( num_elements,     1  ) );
        e.addTestCase( make_shared< TestCaseRadixSort_boost_sample_sort  <T> > ( num_elements,     4  ) );
        e.addTestCase( make_shared< TestCaseRadixSort_boost_sample_sort  <T> > ( num_elements,    16  ) );
        e.addTestCase( make_shared< TestCaseRadixSort_boost_sample_sort  <T> > ( num_elements,    64  ) );
        e.addTestCase( make_shared< TestCaseRadixSort_Metal              <T> > ( num_elements , false, true  , false, num_threads_per_threadgroup ) );
        e.addTestCase( make_shared< TestCaseRadixSort_Metal              <T> > ( num_elements , true,  true  , false, num_threads_per_threadgroup ) );
        e.addTestCase( make_shared< TestCaseRadixSort_Metal              <T> > ( num_elements , false, false , false, num_threads_per_threadgroup ) );
        e.addTestCase( make_shared< TestCaseRadixSort_Metal              <T> > ( num_elements , true,  false , false, num_threads_per_threadgroup ) );
#if TARGET_OS_OSX
        e.addTestCase( make_shared< TestCaseRadixSort_Metal              <T> > ( num_elements , false, false , true , num_threads_per_threadgroup ) );
        e.addTestCase( make_shared< TestCaseRadixSort_Metal              <T> > ( num_elements , true,  false , true , num_threads_per_threadgroup ) );
#endif
        e.execute();
    }
}

#if TARGET_OS_OSX
int main(int argc, char* argv[]) {
#else
int run_test() {
#endif
    TestCaseWithTimeMeasurements::printHeader( cout );

    cerr << "\n\nTesting for type int.\n\n";

    testSuitePerType<int> ();

    cerr << "\n\nTesting for type float.\n\n";

    testSuitePerType<float> ();

    return 0;
}
