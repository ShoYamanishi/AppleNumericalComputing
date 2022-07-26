#include "test_pattern_generation.h"

#include "test_case_cholesky.h"
#include "test_case_cholesky_baseline.h"
#if TARGET_OS_OSX
#include "test_case_cholesky_eigen3.h"
#include "test_case_cholesky_gsl.h"
#endif
#include "test_case_cholesky_lapack.h"
#include "test_case_cholesky_lapack_inverse.h"
#include "test_case_cholesky_metal.h"


template <class T, bool IS_COL_MAJOR>
class TestExecutorCholesky : public TestExecutor {

  protected:

    // Ax = b A is PD and lower triangular, i.e. the upper part is missing.

    const int             m_dim;
    default_random_engine m_e;

    T*                    m_A;
    T*                    m_b;
    T*                    m_L_baseline;
    T*                    m_x_baseline;

  public:

    TestExecutorCholesky(
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
        ,m_A            ( new T [ (dim + 1) * dim / 2 ] )
        ,m_b            ( new T [ dim ]                 )
        ,m_L_baseline   ( new T [ (dim + 1) * dim / 2 ] )
        ,m_x_baseline   ( new T [ dim ]                 )
    {
        generateRandomPDLowerMat<T, IS_COL_MAJOR>( m_A, m_dim, condition_num, m_e );

        fillArrayWithRandomValues<T>( m_e, m_b, m_dim, val_low, val_high );
    }

    virtual ~TestExecutorCholesky()
    {
        delete[] m_A;
        delete[] m_b;
        delete[] m_L_baseline;
        delete[] m_x_baseline;
    }

    void prepareForRun ( const int test_case, const int num ) {

        auto t = dynamic_pointer_cast< TestCaseCholesky<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_A, m_b );
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCaseCholesky<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            memcpy( m_L_baseline, t->getOutputPointer_L(), sizeof(T) * (m_dim+1) * m_dim / 2 );
            memcpy( m_x_baseline, t->getOutputPointer_x(), sizeof(T) * m_dim                 );
        }

        t->compareTruth( m_L_baseline, m_x_baseline );
    }
};

static const size_t NUM_TRIALS = 10;

int matrix_dims[]={ 64, 128, 256, 512, 1024, 2048, 4096 };

template<class T, bool IS_COL_MAJOR>
void testSuitePerType ( const T condition_num, const T gen_low, const T gen_high ) {

    for( auto& dim : matrix_dims ) {
      
        TestExecutorCholesky<T, IS_COL_MAJOR> e( cout, dim, condition_num, gen_low, gen_high, NUM_TRIALS, false );

        e.addTestCase( make_shared< TestCaseCholesky_baseline  <T, IS_COL_MAJOR> > ( dim, false ) );
        e.addTestCase( make_shared< TestCaseCholesky_baseline  <T, IS_COL_MAJOR> > ( dim, true  ) );

        if constexpr (IS_COL_MAJOR) {
#if TARGET_OS_OSX
            e.addTestCase( make_shared< TestCaseCholesky_eigen3  <T, IS_COL_MAJOR> > ( dim ) );
#endif
            e.addTestCase( make_shared< TestCaseCholesky_lapack  <T, IS_COL_MAJOR> > ( dim ) );
            e.addTestCase( make_shared< TestCaseCholesky_lapack_inverse  <T, IS_COL_MAJOR> > ( dim ) );

            if constexpr ( std::is_same< float,T >::value ) {

                e.addTestCase( make_shared< TestCaseCholesky_metal   <T, IS_COL_MAJOR> > ( dim, true  ) );
#if TARGET_OS_OSX
                e.addTestCase( make_shared< TestCaseCholesky_metal   <T, IS_COL_MAJOR> > ( dim, false ) );
#else
                if ( dim <= 1024 ) {
                    e.addTestCase( make_shared< TestCaseCholesky_metal   <T, IS_COL_MAJOR> > ( dim, false ) );
                }
#endif
            }
#if TARGET_OS_OSX
            if constexpr ( std::is_same< double,T >::value ) {

                e.addTestCase( make_shared< TestCaseCholesky_gsl <T, IS_COL_MAJOR> > ( dim ) );
            }
#endif
        }
        e.execute();
    }
}

#if TARGET_OS_OSX
int main( int argc, char* argv[] )
#else
int run_test()
#endif
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

