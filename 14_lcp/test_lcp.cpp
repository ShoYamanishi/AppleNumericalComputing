#include "test_lcp_pattern_generator.h"
#include "test_case_lcp.h"
#include "test_case_lcp_lemke_baseline.h"
#include "test_case_lcp_lemke_vdsp_multithread.h"
#include "test_case_lcp_lemke_neon_multithread.h"
#include "test_case_lcp_lemke_neon.h"
#include "test_case_lcp_lemke_vdsp.h"
#include "test_case_lcp_lemke_bullet.h"
#include "test_case_lcp_pgs.h"
#include "test_case_lcp_pgs_sm.h"

#include <sstream>
#include <algorithm>

template <class T, bool IS_COL_MAJOR>
class TestExecutorLCP : public TestExecutor {

  protected:

    LCPPatternGenerator< T, IS_COL_MAJOR> m_generator;
    const int                             m_dim;
    T*                                    m_M;
    T*                                    m_q;

  public:

    TestExecutorLCP(
        ostream&                 os,
        const int                dim,
        const T                  condition_num,
        const T                  val_low,
        const T                  val_high,
        const int                num_trials,
        const bool               repeatable,
        const LCPTestPatternType p_type,
        const std::string        test_pattern_path
    )
        :TestExecutor   ( os, num_trials )
        ,m_generator    ( repeatable, val_low, val_high, test_pattern_path )
        ,m_dim          ( dim )
        ,m_M            ( new T [ dim * dim ] )
        ,m_q            ( new T [ dim ]       )
    {
        m_generator.generateTestPattern( m_dim, m_M, m_q, p_type, condition_num );
    }

    virtual ~TestExecutorLCP()
    {
        delete[] m_M;
        delete[] m_q;
    }

    void prepareForRun ( const int test_case, const int num ) {

        auto t = dynamic_pointer_cast< TestCaseLCP<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_M, m_q );
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCaseLCP<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        t->checkOutput();
    }
};

static const size_t NUM_TRIALS         = 10;
static const int    MAX_NUM_PIVOTS     = 10000;
static const int    MAX_NUM_ITERATIONS = 10000;
static const float  EPSILON            = 1.0e-15;
static const bool   REPEATABLE         = false;
static const int    NUM_PGS_PER_SM     = 10;
static const float  OMEGA              = 1.0;

int matrix_dims[]={ 64, 128, 256, 512, 1024 };

//int matrix_dims[]={ 64, 128, 256, 512 };

template<class T, bool IS_COL_MAJOR>
void testSuitePerType ( const T condition_num, const T gen_low, const T gen_high, const LCPTestPatternType p_type, const std::string test_pattern_path ) {

    for( auto& dim : matrix_dims ) {
      
        TestExecutorLCP<T, IS_COL_MAJOR> e( cout, dim, condition_num, gen_low, gen_high, NUM_TRIALS, REPEATABLE, p_type, test_pattern_path );

        if constexpr ( !IS_COL_MAJOR ) {

            e.addTestCase( make_shared< TestCaseLCP_lemke_baseline        <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_neon            <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_vdsp            <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_neon_multithread     <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON,  4, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_neon_multithread     <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON,  8, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_neon_multithread     <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON, 16, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_vdsp_multithread     <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON,  4, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_vdsp_multithread     <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON,  8, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_vdsp_multithread     <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_PIVOTS,     (T)EPSILON, 16, p_type ) );

            if constexpr( is_same<float, T>::value ) {
                e.addTestCase( make_shared< TestCaseLCP_lemke_bullet          <T, IS_COL_MAJOR> > ( dim, condition_num, dim * 2,            (T)EPSILON, false, p_type ) );
                e.addTestCase( make_shared< TestCaseLCP_lemke_bullet          <T, IS_COL_MAJOR> > ( dim, condition_num, dim * 2,            (T)EPSILON, true,  p_type ) );
            }
            e.addTestCase( make_shared< TestCaseLCP_pgs                   <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_ITERATIONS, (T)EPSILON, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_pgs_sm                <T, IS_COL_MAJOR> > ( dim, condition_num, MAX_NUM_ITERATIONS, (T)EPSILON, NUM_PGS_PER_SM, p_type ) );
        }
        e.execute();
    }
}

#if TARGET_OS_OSX
int main( int argc, char* argv[] )
{
    const std::string test_pattern_path( "../test_patterns/" );
#else
int run_test( const char* test_pattern_file_path_with_file )
{
    std::string work_file_path( test_pattern_file_path_with_file );

    const std::string test_pattern_path = work_file_path.substr(0, work_file_path.find_last_of("/")) + "/";

#endif
    
    std::stringstream ss;    
    TestCaseWithTimeMeasurements::printHeader( ss );
    auto ss_str =  ss.str();
    ss_str.erase( std::remove_if( ss_str.begin(), ss_str.end(), [](auto ch) { return (ch == '\n' || ch == '\r');  }), ss_str.end() );
    cout << ss_str << "\tfeasibility error\tapprox condition num\tnum pivots\tnum iterations\tnum susspace minimizations\n";

    cerr << "\n\nTesting for type float in row-major with random dinagonally dominant symmetric test patterns.\n\n";

    testSuitePerType<float,  false  > (      1.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > (     10.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > (    100.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > (   1000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > (  10000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > ( 100000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );

    cerr << "\n\nTesting for type double in row-major with random dinagonally dominant symmetric test patterns.\n\n";

    testSuitePerType<double,  false > (      1.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > (     10.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > (    100.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > (   1000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > (  10000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > ( 100000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );

    cerr << "\n\nTesting for type float in row-major with random dinagonally dominant skewsymmetric test patterns.\n\n";

    testSuitePerType<float,  false  > (      1.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > (     10.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > (    100.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > (   1000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > (  10000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<float,  false  > ( 100000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );

    cerr << "\n\nTesting for type double in row-major with random dinagonally dominant skewsymmetric test patterns.\n\n";

    testSuitePerType<double,  false > (      1.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > (     10.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > (    100.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > (   1000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > (  10000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    testSuitePerType<double,  false > ( 100000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );

    cerr << "\n\nTesting for type float in row-major with real nonsymmetric test patterns with mu=0.2.\n\n";

    testSuitePerType<float,  false  > ( 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU02, test_pattern_path );

    cerr << "\n\nTesting for type double in row-major with real nonsymmetric test patterns with mu=0.2.\n\n";

    testSuitePerType<double,  false > ( 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU02, test_pattern_path );

    cerr << "\n\nTesting for type float in row-major with real nonsymmetric test patterns with mu=0.8.\n\n";

    testSuitePerType<float,  false  > ( 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU08, test_pattern_path );

    cerr << "\n\nTesting for type double in row-major with real nonsymmetric test patterns with mu=0.8.\n\n";

    testSuitePerType<double,  false > ( 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU08, test_pattern_path );

    cerr << "\n\nTesting for type float in row-major with real symmetric test patterns.\n\n";

    testSuitePerType<float,  false  > ( 0.0, -1.0, 1.0, LCP_REAL_SYMMETRIC, test_pattern_path );

    cerr << "\n\nTesting for type double in row-major with real symmetric test patterns.\n\n";

    testSuitePerType<double,  false > ( 0.0, -1.0, 1.0, LCP_REAL_SYMMETRIC, test_pattern_path );

    return 0;
}

