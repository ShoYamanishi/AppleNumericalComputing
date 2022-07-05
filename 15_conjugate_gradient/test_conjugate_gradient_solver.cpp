#include <Accelerate/Accelerate.h>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"
#include "conjugate_gradient_metal_cpp.h"

template< class T, bool IS_COL_MAJOR >
class TestCaseConjugateGradientSolver : public TestCaseWithTimeMeasurements  {

  protected:

    const int           m_dim;
    const int           m_max_iteration;
    int                 m_iterations;
    T*                  m_A;
    T*                  m_b;
    T*                  m_x;
    T*                  m_Ap;
    T*                  m_r;
    T*                  m_p;
    const T             m_epsilon;
    const int           m_condition_num;
  public:

    TestCaseConjugateGradientSolver( const int dim, const int max_iteration, const T epsilon, const int condition_num )
        :m_dim          ( dim           )
        ,m_max_iteration( max_iteration )
        ,m_iterations   ( 0             )
        ,m_A            ( nullptr       )
        ,m_b            ( nullptr       )
        ,m_x            ( nullptr       )
        ,m_Ap           ( nullptr       )
        ,m_r            ( nullptr       )
        ,m_p            ( nullptr       )
        ,m_epsilon      ( epsilon       )
        ,m_condition_num( condition_num )
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
            assert(true);
        }
        else {
            setMatrixRowMajor( dim, dim );
        }

        setVerificationType( RMS );

        m_x  = new T[ dim ];
        m_Ap = new T[ dim ];
        m_r  = new T[ dim ];
        m_p  = new T[ dim ];
    }

    virtual ~TestCaseConjugateGradientSolver(){
        delete[] m_x;
        delete[] m_Ap;
        delete[] m_r;
        delete[] m_p;
    }

    virtual void compareTruth()
    {
        // calculate Ax and compare it with b.
        for ( int i = 0; i < m_dim ; i++ ) {
            m_Ap[i] = 0.0;
            for ( int j = 0; j < m_dim ; j++ ) {
                m_Ap[i] += (m_A[i * m_dim + j] * m_x[j]);
            }
        }

        auto rms = getRMSDiffTwoVectors( m_Ap, m_b, m_dim );
        this->setRMS( rms );
    }

    virtual void setInitialStates( T* A,T* b )
    {
        m_A = A;
        m_b = b;
    }

    virtual void run() = 0;

    virtual void printExtra(ostream& os) {
        os << "\t"
           << m_iterations
           << "\t"
           << m_condition_num;
    }

};


template< class T, bool IS_COL_MAJOR >
class TestCaseConjugateGradientSolver_baseline : public TestCaseConjugateGradientSolver< T, IS_COL_MAJOR > {

  public:

    TestCaseConjugateGradientSolver_baseline( const int dim, const int max_iteration, const T epsilon, const int condition_num )
        :TestCaseConjugateGradientSolver< T, IS_COL_MAJOR >( dim, max_iteration, epsilon, condition_num )
    {
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseConjugateGradientSolver_baseline() { }

    virtual void run() {

        memset( this->m_x, 0, sizeof(T) * this->m_dim );

        for ( int i = 0; i < this->m_dim ; i++ ) {

            this->m_Ap[i] = 0.0;

            for ( int j = 0; j < this->m_dim ; j++ ) {

                this->m_Ap[i] += ( this->m_A[ i * this->m_dim + j ] * this->m_x[j] );
            }
            this->m_r[i] = this->m_b[i] - this->m_Ap[i];
            this->m_p[i] = this->m_r[i];
        }

        T max_abs_r = 0.0;

        for ( int i = 0; i < this->m_dim ; i++ ) {

            max_abs_r = max( max_abs_r , fabs(this->m_r[i]) );
        }
        
        if ( max_abs_r < this->m_epsilon ) {
            this->m_iterations = 0;
            return;
        }

        for( this->m_iterations = 1; this->m_iterations <= this->m_max_iteration; this->m_iterations++ ) {

            for ( int i = 0; i < this->m_dim ; i++ ) {

                this->m_Ap[i] = 0.0;

                for ( int j = 0; j < this->m_dim ; j++ ) {

                    this->m_Ap[i] += ( this->m_A[ i * this->m_dim + j ] * this->m_p[j] );
                }
            }

            T rtr = 0.0;
            T pAp = 0.0;

            for ( int i = 0; i < this->m_dim ; i++ ) {

                rtr += ( this->m_r[i] * this->m_r[i]  );
                pAp += ( this->m_p[i] * this->m_Ap[i] );

            }

            const T alpha = rtr / pAp;

            T rtr2 = 0.0;

            for ( int i = 0; i < this->m_dim ; i++ ) {

                this->m_x[i] = this->m_x[i] + alpha * this->m_p[i];
                this->m_r[i] = this->m_r[i] - alpha * this->m_Ap[i];

                rtr2 += ( this->m_r[i] * this->m_r[i] );
            }

            max_abs_r = 0.0;
            for ( int i = 0; i < this->m_dim ; i++ ) {

                max_abs_r = max( max_abs_r , fabs(this->m_r[i]) );
            }
        
            if ( max_abs_r < this->m_epsilon ) {
                return;
            }

            const T beta = rtr2 / rtr;

            for ( int i = 0; i < this->m_dim ; i++ ) {

                this->m_p[i] = this->m_r[i] + beta * this->m_p[i];
            }
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseConjugateGradientSolver_vdsp: public TestCaseConjugateGradientSolver< T, IS_COL_MAJOR > {

  public:

    TestCaseConjugateGradientSolver_vdsp( const int dim, const int max_iteration, const T epsilon, const int condition_num )
        :TestCaseConjugateGradientSolver< T, IS_COL_MAJOR >( dim, max_iteration, epsilon, condition_num )
    {
        this->setImplementationType( VDSP );
    }

    virtual ~TestCaseConjugateGradientSolver_vdsp() { }

    virtual void run() {

        memset( this->m_x, 0, sizeof(T) * this->m_dim );

        if constexpr ( is_same< float,T >::value ) {
            vDSP_mmul( this->m_A,  1, this->m_x, 1, this->m_Ap, 1, this->m_dim, 1, this->m_dim );
            vDSP_vsub( this->m_Ap, 1, this->m_b, 1, this->m_r,  1, this->m_dim );
        }
        else {
            vDSP_mmulD( this->m_A,  1, this->m_x, 1, this->m_Ap, 1, this->m_dim, 1, this->m_dim );
            vDSP_vsubD( this->m_Ap, 1, this->m_b, 1, this->m_r,  1, this->m_dim );
        }

        memcpy( this->m_p, this->m_r, sizeof(T) * this->m_dim );

        T min_r = 0.0;
        T max_r = 0.0;

        if constexpr ( is_same< float,T >::value ) {
            vDSP_minv( this->m_r, 1, &min_r, this->m_dim );
            vDSP_maxv( this->m_r, 1, &min_r, this->m_dim );
        }
        else {
            vDSP_minvD( this->m_r, 1, &min_r, this->m_dim );
            vDSP_maxvD( this->m_r, 1, &min_r, this->m_dim );
        }

        if ( fabs(min_r) < this->m_epsilon && fabs(max_r) < this->m_epsilon  ) {
            this->m_iterations = 0;
            return;
        }

        for( this->m_iterations = 1; this->m_iterations <= this->m_max_iteration; this->m_iterations++ ) {

            T rtr = 0.0;
            T pAp = 0.0;

            if constexpr ( is_same< float,T >::value ) {

                vDSP_mmul( this->m_A,  1, this->m_p, 1, this->m_Ap, 1, this->m_dim, 1, this->m_dim );
                vDSP_dotpr( this->m_r, 1, this->m_r,  1, &rtr, this->m_dim );
                vDSP_dotpr( this->m_p, 1, this->m_Ap, 1, &pAp, this->m_dim );

            }
            else {
                vDSP_mmulD( this->m_A,  1, this->m_p, 1, this->m_Ap, 1, this->m_dim, 1, this->m_dim );
                vDSP_dotprD( this->m_r, 1, this->m_r,  1, &rtr, this->m_dim );
                vDSP_dotprD( this->m_p, 1, this->m_Ap, 1, &pAp, this->m_dim );
            }

            const T alpha  = rtr / pAp;
            const T malpha = -1.0 * alpha;

            if constexpr ( is_same< float,T >::value ) {

                vDSP_vsma( this->m_p,  1,  &alpha, this->m_x, 1, this->m_x, 1, this->m_dim );
                vDSP_vsma( this->m_Ap, 1, &malpha, this->m_r, 1, this->m_r, 1, this->m_dim );
            }
            else {
                vDSP_vsmaD( this->m_p,  1 , &alpha, this->m_x, 1, this->m_x, 1, this->m_dim );
                vDSP_vsmaD( this->m_Ap, 1, &malpha, this->m_r, 1, this->m_r, 1, this->m_dim );
            }

            T rtr2 = 0.0;
            if constexpr ( is_same< float,T >::value ) {
                vDSP_dotpr( this->m_r, 1, this->m_r, 1, &rtr2, this->m_dim );
            }
            else {
                vDSP_dotprD( this->m_r, 1, this->m_r, 1, &rtr2, this->m_dim );
            }


            if constexpr ( is_same< float,T >::value ) {
                vDSP_minv( this->m_r, 1, &min_r, this->m_dim );
                vDSP_maxv( this->m_r, 1, &min_r, this->m_dim );
            }
            else {
                vDSP_minvD( this->m_r, 1, &min_r, this->m_dim );
                vDSP_maxvD( this->m_r, 1, &min_r, this->m_dim );
            }

            if ( fabs(min_r) < this->m_epsilon && fabs(max_r) < this->m_epsilon  ) {
                return;
            }

            const T beta = rtr2 / rtr;

            if constexpr ( is_same< float,T >::value ) {

                vDSP_vsma( this->m_p, 1, &beta, this->m_r, 1, this->m_p, 1, this->m_dim );
            }
            else {
                vDSP_vsmaD( this->m_p, 1, &beta, this->m_r, 1, this->m_p, 1, this->m_dim );
            }
        }
    }
};


template<class T, bool IS_COL_MAJOR>
class TestCaseConjugateGradientSolver_metal : public TestCaseConjugateGradientSolver<T, IS_COL_MAJOR> {

  private:
    ConjugateGradientMetalCpp m_metal;

  public:

    TestCaseConjugateGradientSolver_metal( const int dim, const int max_num_iterations, const T epsilon, const int condition_num, const int num_threads_per_group )
        :TestCaseConjugateGradientSolver< T, IS_COL_MAJOR >( dim, max_num_iterations, epsilon, condition_num )
        ,m_metal( dim, num_threads_per_group, max_num_iterations, epsilon )
    {
        static_assert( std::is_same<float, T>::value );

        this->setMetal( ONE_PASS_ATOMIC_SIMD_SUM, 1, num_threads_per_group );
    }

    virtual ~TestCaseConjugateGradientSolver_metal(){;}

    virtual void setInitialStates( T* A,T* b )
    {
        TestCaseConjugateGradientSolver<T, IS_COL_MAJOR>::setInitialStates( A, b );

        memcpy( m_metal.getRawPointerA(), A, sizeof(float) * this->m_dim * this->m_dim );
        memcpy( m_metal.getRawPointerB(), b, sizeof(float) * this->m_dim );
    }

    void run() {

        m_metal.performComputation();
        memcpy( this->m_x, m_metal.getX(), sizeof(float) * this->m_dim );
        this->m_iterations = m_metal.getNumIterations();
    }
};


template<class T, bool IS_COL_MAJOR>
class TestExecutorConjugateGradientSolver : public TestExecutor {

  protected:
    const int             m_dim;
    default_random_engine m_e;
    T*                    m_A;
    T*                    m_b;

  public:

    TestExecutorConjugateGradientSolver (
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
        ,m_b            ( new T [ dim ]       )
    {
        generateRandomPDMat<T, IS_COL_MAJOR>( m_A, m_dim, condition_num, m_e );
        fillArrayWithRandomValues( m_e, m_b, m_dim, val_low, val_high );
    }

    void prepareForRun ( const int test_case, const int num ) {

        auto t = dynamic_pointer_cast< TestCaseConjugateGradientSolver<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        t->setInitialStates( m_A, m_b );
    }

    void cleanupAfterBatchRuns ( const int test_case ) {

        auto t = dynamic_pointer_cast< TestCaseConjugateGradientSolver<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        t->compareTruth();
    }

    virtual ~TestExecutorConjugateGradientSolver()
    {
        delete[] m_A;
        delete[] m_b;
    }
};

static const size_t NUM_TRIALS    = 10;
static const int    MAX_ITERATION = 1000;
static const double EPSILON       = 1.0e-8;
static const int    METAL_NUM_THREADS_PER_THREADGROUP = 1024;

#if TARGET_OS_OSX
int  matrix_dims[]={ 64, 128, 256, 512, 1024, 2048, 4096 };
#else
int  matrix_dims[]={ 64, 128, 256, 512, 1024 };
#endif

template<class T, bool IS_COL_MAJOR>
void testSuitePerType ( const T condition_num, const T gen_low, const T gen_high ) {

    const int neon_num_lanes = ( is_same<float, T>::value )? 4 : 2;

    for( auto& dim : matrix_dims ) {

        TestExecutorConjugateGradientSolver<T, IS_COL_MAJOR> e( cout, dim, condition_num, gen_low, gen_high, NUM_TRIALS, false );

        e.addTestCase( make_shared< TestCaseConjugateGradientSolver_baseline    <T, IS_COL_MAJOR> > ( dim, MAX_ITERATION, (const T)EPSILON, condition_num ) );
        e.addTestCase( make_shared< TestCaseConjugateGradientSolver_vdsp        <T, IS_COL_MAJOR> > ( dim, MAX_ITERATION, (const T)EPSILON, condition_num ) );
        if constexpr ( is_same<float, T>::value ) {
            e.addTestCase( make_shared< TestCaseConjugateGradientSolver_metal       <T, IS_COL_MAJOR> > ( dim, MAX_ITERATION, (const T)EPSILON, condition_num, METAL_NUM_THREADS_PER_THREADGROUP ) );
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
    std::stringstream ss;    
    TestCaseWithTimeMeasurements::printHeader( ss );
    auto ss_str =  ss.str();
    ss_str.erase( std::remove_if( ss_str.begin(), ss_str.end(), [](auto ch) { return (ch == '\n' || ch == '\r');  }), ss_str.end() );
    cout << ss_str << "\tnum iterations\tcondition num\n";

    cerr << "\n\nTesting for type float in row-major.\n\n";

    testSuitePerType<float, false > ( 1.0, -1.0, 1.0 );
    testSuitePerType<float, false > ( 10.0, -1.0, 1.0 );
    testSuitePerType<float, false > ( 100.0, -1.0, 1.0 );
    testSuitePerType<float, false > ( 1000.0, -1.0, 1.0 );
    testSuitePerType<float, false > ( 10000.0, -1.0, 1.0 );
    testSuitePerType<float, false > ( 100000.0, -1.0, 1.0 );

    cerr << "\n\nTesting for type double in row-major.\n\n";

    testSuitePerType<double, false > ( 1.0, -1.0, 1.0 );
    testSuitePerType<double, false > ( 10.0, -1.0, 1.0 );
    testSuitePerType<double, false > ( 100.0, -1.0, 1.0 );
    testSuitePerType<double, false > ( 1000.0, -1.0, 1.0 );
    testSuitePerType<double, false > ( 10000.0, -1.0, 1.0 );
    testSuitePerType<double, false > ( 100000.0, -1.0, 1.0 );

    return 0;
}
