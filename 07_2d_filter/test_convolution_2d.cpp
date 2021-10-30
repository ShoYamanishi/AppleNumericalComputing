#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"

#include "convolution_2d_ciimage_cpp.h"
#include "convolution_2d_metal_cpp.h"

template<class T>
class TestCaseConvolution2D :public TestCaseWithTimeMeasurements {

  protected:
    const size_t        m_image_width;
    const size_t        m_image_height;
    const T*            m_image_in;
    const T*            m_kernel;
    T*                  m_image_out;

  public:

    TestCaseConvolution2D( const size_t width, const size_t height )
        :m_image_width           ( width     )
        ,m_image_height          ( height    )
        ,m_image_in              ( nullptr   )
        ,m_kernel                ( nullptr   )
        ,m_image_out             ( nullptr   )
    {
        if constexpr ( is_same<float, T>::value ) {

            setDataElementType( FLOAT );
        }
        else {
            assert(true);
        }

        setMatrixRowMajor( height, width );

        setVerificationType( RMS );
    }

    virtual ~TestCaseConvolution2D() {;}

    virtual void compareTruth( const T* const baseline ) {

        auto rms = getRMSDiffTwoVectors( getImageOut(), baseline, this->m_image_width * this->m_image_height );
        setRMS( rms ) ;
    }

    virtual void setInitialStates( const T* const image_in, const T* kernel, T* image_out ) {
        m_image_in  = image_in;
        m_kernel    = kernel;
        m_image_out = image_out;
    }

    virtual void run() = 0;

    virtual T* getImageOut() { return m_image_out; }
};


// Calculate one NxN convolution.
template<class T, int KERNEL_DIM>
static inline T calc_conv_one_point (
    const T* const kernel,
    const T* const in, 
    const size_t   image_width,
    const size_t   image_height,
    const size_t   center_x,
    const size_t   center_y
) {
    static_assert( KERNEL_DIM % 2 == 1 );// DIM must be odd.
    
    constexpr int KERN_OFFSET = KERNEL_DIM/2;

    T sum = 0;

    for ( int kern_y = 0; kern_y < KERNEL_DIM; kern_y++ ) {

        const int image_y = ( kern_y - KERN_OFFSET ) + center_y;

        if ( 0 <= image_y && image_y < image_height ) {

            for ( int kern_x = 0; kern_x < KERNEL_DIM; kern_x++ ) {

                const int image_x = ( kern_x - KERN_OFFSET ) + center_x;

                const T v =   ( 0 <= image_x && image_x < image_width )
                            ? ( kernel[ index_row_major(kern_x, kern_y, KERNEL_DIM) ] * in[ index_row_major(image_x, image_y, image_width) ] )
                            : 0
                            ;
                sum += v;
            }
        }
    }
    return sum;
}


// Calculate NxN convolution.
template<class T, int KERNEL_DIM>
static inline void calc_conv_row_block (

    const T* const kernel,
    const T* const in, 
          T* const out, 
    const size_t   width,
    const size_t   height,
    const size_t   row_start,
    const size_t   row_end_past_one

) {
    for ( size_t y = row_start; y < row_end_past_one; y++ ) {

        for ( size_t x = 0; x < width; x++ ) {

            out[ index_row_major( x, y, width) ] = calc_conv_one_point< T,KERNEL_DIM >( kernel, in, width, height, x, y );
        }
    }
}


template< class T, int KERNEL_DIM >
class TestCaseConvolution2D_baseline : public TestCaseConvolution2D<T> {

  public:
    TestCaseConvolution2D_baseline( const size_t width, const size_t height )
        :TestCaseConvolution2D<T>( width, height )
    {
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseConvolution2D_baseline(){;}

    virtual void run(){

        calc_conv_row_block<T,KERNEL_DIM> (
            this->m_kernel,
            this->m_image_in,
            this->m_image_out,
            this->m_image_width,
            this->m_image_height,
            0,
            this->m_image_height
        );
    }
};


template< class T, int KERNEL_DIM >
class TestCaseConvolution2D_multithreads : public TestCaseConvolution2D<T> {

    const size_t                m_num_threads;

    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    vector<thread>              m_threads;

  public:
    TestCaseConvolution2D_multithreads( const size_t width, const size_t height, const size_t num_threads )
        :TestCaseConvolution2D<T>( width, height )
        ,m_num_threads ( num_threads )
        ,m_fan_out     ( num_threads )
        ,m_fan_in      ( num_threads )
    {
        this->setCPPBlock( num_threads, 1 );

        const size_t num_rows_per_thread = this->m_image_height / m_num_threads;

        auto thread_lambda = [ this, num_rows_per_thread ]( const size_t thread_index ) {

            while ( true ) {

                const size_t row_begin = thread_index * num_rows_per_thread;
                const size_t row_end   = row_begin + num_rows_per_thread;

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                calc_conv_row_block<T,KERNEL_DIM> (
                    this->m_kernel,
                    this->m_image_in,
                    this->m_image_out,
                    this->m_image_width,
                    this->m_image_height,
                    row_begin,
                    row_end
                );

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

    virtual ~TestCaseConvolution2D_multithreads() {

        m_fan_out.terminate();
        m_fan_in. terminate();

        for ( auto& t : m_threads ) {
            t.join();
        }
    }

    virtual void run(){

        m_fan_out.notify();
        m_fan_in. wait();
    }
};


template<class T, int KERNEL_DIM>
class TestCaseConvolution2D_ciimage : public TestCaseConvolution2D<T> {

    Convolution2D_CIImageCpp m_ci_image;

  public:
    TestCaseConvolution2D_ciimage( const size_t width, const size_t height, const bool use_gpu )
        :TestCaseConvolution2D<T>( width, height )
        ,m_ci_image( width, height, KERNEL_DIM, use_gpu )
    {
        static_assert( is_same< float,T >::value );
        this->setImplementationType( use_gpu ? CIIMAGE_CPU : CIIMAGE_GPU );
    }

    virtual ~TestCaseConvolution2D_ciimage(){;}

    virtual void run() {
        m_ci_image.performConvolution();
    }

    virtual void setInitialStates( const T* const image_in, const T* kernel, T* image_out ) {
        this->m_image_in  = image_in;
        this->m_kernel    = kernel;
        this->m_image_out = image_out;
        m_ci_image.copyToInputBuffer ( image_in );
        m_ci_image.copyToKernelBuffer( kernel   );
    }

    virtual T* getImageOut() {

        auto* p = m_ci_image.getOutputImagePtr();
        memcpy( this->m_image_out, p, sizeof(T) * (this->m_image_width) * (this->m_image_height) );
        return p;
    }
};


template<class T, int KERNEL_DIM>
class TestCaseConvolution2D_metal : public TestCaseConvolution2D<T> {

  protected:
    Convolution2DMetalCpp m_metal;

  public:
    TestCaseConvolution2D_metal( const size_t width, const size_t height, const int algo_type )
        :TestCaseConvolution2D<T>( width, height )
        ,m_metal( width, height, KERNEL_DIM, algo_type )
    {
        static_assert( is_same< float,T >::value );
        if (algo_type == 0 ) {
            this->setMetal( NAIVE, 1024, 1024 );
        }
        else if (algo_type == 1 ) {
            this->setMetal( TWO_STAGES, 1024, 1024 );
        }
        else {
            this->setMetal( MPS, 1024, 1024 );
        }
    }

    virtual ~TestCaseConvolution2D_metal(){;}

    virtual void run() {

        m_metal.performConvolution();
    }

    virtual void setInitialStates( const T* const image_in, const T* kernel, T* image_out ) {

        this->m_image_in  = image_in;
        this->m_kernel    = kernel;
        this->m_image_out = image_out;
        m_metal.copyToInputBuffer ( image_in );
        m_metal.copyToKernelBuffer( kernel );
    }

    virtual T* getImageOut() {
        return  m_metal.getOutputImagePtr();
    }
};


template <class T, int KERNEL_DIM>
class TestExecutorConvolution2D : public TestExecutor {

  protected:

    const int             m_image_width;
    const int             m_image_height;
    const int             m_num_pixels;
    const bool            m_repeatable;
    default_random_engine m_e;
    T* const              m_image_in;
    T* const              m_image_out;
    T* const              m_image_out_baseline;
    T* const              m_kernel;

  public:

    TestExecutorConvolution2D(
        ostream&   os,
        const int  image_width,
        const int  image_height,
        const int  num_trials,
        const bool repeatable, 
        const T    low,
        const T    high
    )
        :TestExecutor        ( os, num_trials )
        ,m_image_width       ( image_width )
        ,m_image_height      ( image_height )
        ,m_num_pixels        ( image_width * image_height )
        ,m_repeatable        ( repeatable )
        ,m_e                 ( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_image_in          ( new T[ m_num_pixels] )
        ,m_image_out         ( new T[ m_num_pixels] )
        ,m_image_out_baseline( new T[ m_num_pixels] )
        ,m_kernel            ( new T[ KERNEL_DIM * KERNEL_DIM ] )
    {
        fillArrayWithRandomValues( m_e,  m_image_in, m_num_pixels,            low, high );
        fillArrayWithRandomValues( m_e,  m_kernel,   KERNEL_DIM * KERNEL_DIM, low, high );
    }

    virtual ~TestExecutorConvolution2D() {

        delete[] m_image_in;
        delete[] m_image_out;
        delete[] m_image_out_baseline;
        delete[] m_kernel;
    }

    void prepareForRun ( const int test_case, const int num ) {

        memset( m_image_out, 0,  sizeof(T) * m_num_pixels );

        auto t = dynamic_pointer_cast< TestCaseConvolution2D<T> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_image_in, m_kernel, m_image_out );
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCaseConvolution2D<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            memcpy( m_image_out_baseline, m_image_out, sizeof(T) * m_num_pixels );
        }
        t->compareTruth( m_image_out_baseline );
    }

    const string preamble () {
        return "num elems:[" + to_string(m_image_width) + ", " + to_string(m_image_height) +  "]";
    }
};


static const size_t NUM_TRIALS = 10;

struct image_dim {
    size_t width;
    size_t height;
};

struct image_dim image_dims[]={
    {       64,      64}, 
    {      128,     128}, 
    {      256,     256}, 
    {      512,     512}, 
    {     1024,    1024}, 
    {   2*1024,  2*1024}, 
    {   4*1024,  4*1024}, 
    {   8*1024,  8*1024}, 
    {  16*1024, 16*1024}, 
    {  32*1024, 32*1024}
};

template<class T, int KERNEL_DIM>
void testSuitePerType ( const T gen_low, const T gen_high ) {

    for( auto dims : image_dims ) {

        const auto w = dims.width;
        const auto h = dims.height;

        TestExecutorConvolution2D<T, KERNEL_DIM> e( cout, w, h, NUM_TRIALS, false, gen_low, gen_high );

        e.addTestCase( make_shared< TestCaseConvolution2D_baseline     <T, KERNEL_DIM> > ( w, h        ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_multithreads <T, KERNEL_DIM> > ( w, h,  2    ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_multithreads <T, KERNEL_DIM> > ( w, h,  4    ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_multithreads <T, KERNEL_DIM> > ( w, h,  8    ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_multithreads <T, KERNEL_DIM> > ( w, h, 16    ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_multithreads <T, KERNEL_DIM> > ( w, h, 32    ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_ciimage      <T, KERNEL_DIM> > ( w, h, false ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_ciimage      <T, KERNEL_DIM> > ( w, h, true  ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_metal        <T, KERNEL_DIM> > ( w, h, 0     ) );
        e.addTestCase( make_shared< TestCaseConvolution2D_metal        <T, KERNEL_DIM> > ( w, h, 1     ) );
        if ( w <= 16*1024 && h <= 16*1024 ) {
            e.addTestCase( make_shared< TestCaseConvolution2D_metal        <T, KERNEL_DIM> > ( w, h, 2     ) );
        }

        e.execute();
    }
}


int main( int argc, char* argv[] )
{
    TestCaseWithTimeMeasurements::printHeader( cout );

    cerr << "\n\nTesting for type float.\n\n";

    testSuitePerType<float, 5> ( -1.0, 1.0 );

    return 0;
}
