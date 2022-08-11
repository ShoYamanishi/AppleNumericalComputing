#include <Accelerate/Accelerate.h>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"
#include <math.h>

template< class T, bool IS_COL_MAJOR >
class TestCaseNonsymmetricBandMat : public TestCaseWithTimeMeasurements  {

  protected:

    const int           m_dim;
    const int           m_band_width;
    T*                  m_A;
    T*                  m_b;
    T*                  m_x;
    const T             m_epsilon;
  public:

    TestCaseNonsymmetricBandMat( const int dim, const T epsilon, const int band_width )
        :m_dim          ( dim           )
        ,m_band_width   ( band_width    )
        ,m_A            ( nullptr       )
        ,m_b            ( nullptr       )
        ,m_x            ( nullptr       )
        ,m_epsilon      ( epsilon       )
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
    }

    virtual ~TestCaseNonsymmetricBandMat() {
        delete[] m_x;
    }

    virtual void compareTruth()
    {
        // calculate Ax and compare it with b.
        T* Ax = new T[m_dim];
        for ( int i = 0; i < m_dim; i++ ) {
            T sum = 0.0;
            for ( int j = 0; j < m_dim; j++ ) {
                sum += ( m_A[linear_index_mat<IS_COL_MAJOR>(i, j, m_dim, m_dim)] * m_x[j] );
            }
            Ax[i] = sum;
        }

        auto rms = getRMSDiffTwoVectors( Ax, m_b, m_dim );
        this->setRMS( rms );
    }

    virtual void setInitialStates( T* A,T* b )
    {
        m_A = A;
        m_b = b;
    }

    virtual void run() = 0;

};


template< class T, bool IS_COL_MAJOR >
class TestCaseNonsymmetricBandMat_lapack : public TestCaseNonsymmetricBandMat< T, IS_COL_MAJOR > {

  private:

    T* m_AB;
    T* m_bx;

    // Format of the Matrix AB \in [ (KL + KL + KU + 1) x DIM ]
    //
    // !!!PLEASE NOTE THAT AB IS IN COL-MAJOR!!!
    //
    //           +---------------------------------------------+
    // 1         |  *    *    *    *    *    *    *    *    *  |
    //           |  *    *    *    *    *    *    *    *    *  |
    // KL        |  *    *    *    *    *    *    *    *    *  |
    //           +=============================================+
    // KL+1      |  *    *    *   a14  a25  a36  a47  a58  a69 |
    //           |  *    *   a13  a24  a35  a46  a57  a68  a79 |
    // KL+KL     |  *   a12  a23  a34  a45  a56  a67  a78  a89 |
    //           +---------------------------------------------+
    // KL+KL+1   | a11  a22  a33  a44  a55  a66  a77  a88  a99 | <= diagonal entries
    //           +---------------------------------------------+
    // KL+KL+1+1 | a21  a32  a43  a54  a65  a76  a87  a97   *  |
    //           | a31  a42  a53  a64  a75  a86  a97   *    *  |
    // KL+KL+1+KU| a41  52   a63  a74  a85  a96   *    *    *  |
    //           +---------------------------------------------+

  public:

    TestCaseNonsymmetricBandMat_lapack( const int dim, const T epsilon, const int band_width )
        :TestCaseNonsymmetricBandMat< T, IS_COL_MAJOR >( dim, epsilon, band_width )
        ,m_AB( new T[ dim * ( band_width * 3 + 1 )] )
        ,m_bx( new T[ dim ] )
    {
        this->setImplementationType( LAPACK );
    }
   
    virtual ~TestCaseNonsymmetricBandMat_lapack() {
        delete[] m_AB;
        delete[] m_bx;
    }

    virtual void setInitialStates( T* A, T* b ) {

        const int AB_diag_row = 2 * this->m_band_width;
        const int row_size = this->m_band_width * 3 + 1 ;
        const int col_size = this->m_dim;
        memset(m_AB, 0, sizeof(T) * col_size * row_size );

        for ( int i = 0; i < this->m_dim; i++ ) {

            for ( int j = max( 0,           i - this->m_band_width     ); 
                      j < min( this->m_dim, i + this->m_band_width + 1 ); j++ ) {

                // m_AB is in col-major.
                const int AB_row = AB_diag_row + (i - j);
                m_AB[ j * row_size + AB_row ] = A[linear_index_mat<IS_COL_MAJOR>(i, j, this->m_dim, this->m_dim)];

            }
            m_bx[i] = b[i];
        }

        TestCaseNonsymmetricBandMat<T,IS_COL_MAJOR>::setInitialStates( A, b );
    }


    virtual void run() {

        int    n       = this->m_dim;
        int    kl      = this->m_band_width;
        int    ku      = this->m_band_width;
        int    nrhs    = 1;
        int    ldab    = this->m_band_width * 3 + 1;
        int*   ipiv    = new int[this->m_dim];
        int    ldb     = this->m_dim;
        int    info;
        int    r = -1;

        if constexpr ( std::is_same< float,T >::value ) {

            r = sgbsv_( &n, &kl, &ku, &nrhs, m_AB, &ldab, ipiv, m_bx, &ldb, &info );
        }
        else {
            r = dgbsv_( &n, &kl, &ku, &nrhs, m_AB, &ldab, ipiv, m_bx, &ldb, &info );
        }
        if ( r != 0 ) {
            std::cerr << "sposv returned non zeror:" << r << " info:"  << info << "\n";
        }

        for ( int i = 0 ; i < this->m_dim; i++ ) {
            const int pivot = ipiv[i] - 1;
            this->m_x[pivot] = m_bx[i];
        }

        delete[] ipiv;
    }
};



template<class T, bool IS_COL_MAJOR>
class TestExecutorNonsymmetricBandMat : public TestExecutor {

  protected:
    const int             m_dim;
    const int             m_band_width;
    default_random_engine m_e;
    T*                    m_A;
    T*                    m_b;

  public:
    TestExecutorNonsymmetricBandMat(
        ostream&   os,
        const int  dim,
        const int  band_width,
        const T    val_low,
        const T    val_high,
        const int  num_trials,
        const bool repeatable
    )
        :TestExecutor   ( os, num_trials )
        ,m_dim          ( dim )
        ,m_band_width   ( band_width )
        ,m_e            ( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_A            ( new T [ dim * dim ] )
        ,m_b            ( new T [ dim ]       )
    {

        generateRandomCopositiveMat<T>( m_A, m_dim, 10.0, m_e );
                
        // make m_A a band matrix.
        for ( int i = 0 ; i < m_dim; i++ ) {
            for ( int j = 0 ; j < i - m_band_width ; j++ ) {
                m_A[linear_index_mat<IS_COL_MAJOR>(i, j, m_dim, m_dim)] = 0.0;
            }

            for ( int j = i + m_band_width + 1; j < m_dim; j++ ) {
                m_A[linear_index_mat<IS_COL_MAJOR>(i, j, m_dim, m_dim)] = 0.0;
            }
        }

        fillArrayWithRandomValues( m_e, m_b, m_dim, val_low, val_high );
    }

    void prepareForRun ( const int test_case, const int num ) {

        auto t = dynamic_pointer_cast< TestCaseNonsymmetricBandMat<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        t->setInitialStates( m_A, m_b );
    }

    void cleanupAfterBatchRuns ( const int test_case ) {

        auto t = dynamic_pointer_cast< TestCaseNonsymmetricBandMat<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        t->compareTruth();
    }

    virtual ~TestExecutorNonsymmetricBandMat()
    {
        delete[] m_A;
        delete[] m_b;
    }
};

static const size_t NUM_TRIALS    = 10;
static const double EPSILON       = 1.0e-8;

#if TARGET_OS_OSX
int  matrix_dims[]={ 64, 128, 256, 512, 1024, 2048, 4096 };

#else
int  matrix_dims[]={ 64, 128, 256, 512, 1024, 2048, 4096 };
#endif


template<class T, bool IS_COL_MAJOR>
void testSuitePerType ( const T gen_low, const T gen_high ) {

    const int neon_num_lanes = ( is_same<float, T>::value )? 4 : 2;

    for( auto& dim : matrix_dims ) {

        const int band_width = (int)(sqrt(dim));

        TestExecutorNonsymmetricBandMat<T, IS_COL_MAJOR> e( cout, dim, band_width, gen_low, gen_high, NUM_TRIALS, false );

        e.addTestCase( make_shared< TestCaseNonsymmetricBandMat_lapack<T, IS_COL_MAJOR> >( dim, (const T)EPSILON, band_width ) );

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
    TestCaseWithTimeMeasurements::printHeader( cout );

    cerr << "\n\nTesting for type float in row-major.\n\n";

    testSuitePerType<float, false > (-1.0, 1.0 );

    cerr << "\n\nTesting for type double in row-major.\n\n";

    testSuitePerType<double, false > ( -1.0, 1.0 );

    return 0;
}
