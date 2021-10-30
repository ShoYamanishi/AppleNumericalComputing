#include <Accelerate/Accelerate.h>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"

#include "dense_matrix_vector_metal_cpp.h"

template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV : public TestCaseWithTimeMeasurements  {

  protected:
    const int           m_M;
    const int           m_N;
    T*                  m_matrix;
    T*                  m_vector;
    T*                  m_output_vector;

  public:

    TestCaseDenseMV( const int M, const int N )
        :m_M             ( M       )
        ,m_N             ( N       )
        ,m_matrix        ( nullptr )
        ,m_vector        ( nullptr )
        ,m_output_vector ( nullptr )
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

        if constexpr (IS_COL_MAJOR) {
            setMatrixColMajor( M, N );
        }
        else {
            setMatrixRowMajor( M, N );
        }

        setVerificationType( RMS );
    }

    virtual ~TestCaseDenseMV() {;}

    virtual void compareTruth( const T* const baseline ) {

        auto rms = getRMSDiffTwoVectors( getOutputVector(), baseline, m_M );
        setRMS( rms );

//        auto tmp = getOutputVector();
//        for (int i = 0; i < m_M; i++) {
//            cerr << "[" << i << "]:" << baseline[i] << "\t" << tmp[i] << "\n";
//        }
    }

    virtual T* getOutputVector() {
        return m_output_vector;
    }

    virtual void setInitialStates( T* matrix, T* vector, T* output_vector ) {
        m_matrix        = matrix;
        m_vector        = vector;
        m_output_vector = output_vector;
    }

    virtual void run() = 0;
};


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_baseline : public TestCaseDenseMV< T, IS_COL_MAJOR > {

  public:

    TestCaseDenseMV_baseline( const int M, const int N )
        :TestCaseDenseMV<T, IS_COL_MAJOR>( M, N )
    {
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseDenseMV_baseline(){;}

    virtual void run() {

        for ( int i = 0; i < this->m_M; i++ ) {

            this->m_output_vector[i] = 0.0;
            for ( int j = 0; j < this->m_N; j++ ) {

                const int mat_index = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );

                this->m_output_vector[i] += ( this->m_matrix[ mat_index ] * this->m_vector[j] );
            }
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_multithread : public TestCaseDenseMV<T, IS_COL_MAJOR> {

    const size_t                m_num_threads;
    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    vector<thread>              m_threads;

  public:
    TestCaseDenseMV_multithread( const int M, const int N, const int num_threads )
        :TestCaseDenseMV<T, IS_COL_MAJOR>( M, N )
        ,m_num_threads( num_threads )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
    {
        assert ( M % 4 == 0 && N % 4 == 0 && M >= 4 && N >= 4 );

        this->setCPPBlock( num_threads, 1 );

        const size_t num_rows_per_thread = this->m_M / m_num_threads;

        auto thread_lambda = [ this, num_rows_per_thread ]( const size_t thread_index ) {

            const size_t row_begin = thread_index * num_rows_per_thread;
            const size_t row_end   = row_begin + num_rows_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                for ( int i = row_begin; i < row_end; i++ ) {

                    this->m_output_vector[i] = 0.0;
                    for ( int j = 0; j < this->m_N; j++ ) {

                        const int mat_index = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );

                        this->m_output_vector[i] += ( this->m_matrix[ mat_index ] * this->m_vector[j] );
                    }
                }

                m_fan_in.notify();
                if( m_fan_in.isTerminating() ) {
                    break;
                }
            }
        };

        for ( size_t i = 0; i < num_threads; i++ ) {
            m_threads.emplace_back( thread_lambda, i );
        }
    }

    virtual ~TestCaseDenseMV_multithread() {

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


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_blas : public TestCaseDenseMV< T, IS_COL_MAJOR > {

  public:

    TestCaseDenseMV_blas( const int M, const int N )
        :TestCaseDenseMV<T, IS_COL_MAJOR>( M, N )
    {
        this->setImplementationType(BLAS);
    }

    virtual ~TestCaseDenseMV_blas(){;}

    virtual void run() {

        CBLAS_ORDER order = IS_COL_MAJOR ? CblasColMajor : CblasRowMajor;

        if constexpr ( is_same< float,T >::value ) {

            cblas_sgemv( order, CblasNoTrans, this->m_M,  this->m_N, 1.0, this->m_matrix, this->m_M, this->m_vector, 1, 1.0, this->m_output_vector, 1 );
        }
        else { // is_same< double,T >::value
            cblas_dgemv( order, CblasNoTrans, this->m_M,  this->m_N, 1.0, this->m_matrix, this->m_M, this->m_vector, 1, 1.0, this->m_output_vector, 1 );
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_vDSP : public TestCaseDenseMV<T, IS_COL_MAJOR> {

  public:

    TestCaseDenseMV_vDSP( const int M, const int N )
        :TestCaseDenseMV<T, IS_COL_MAJOR>( M, N )
    {
        static_assert( !IS_COL_MAJOR );

        this->setImplementationType(VDSP);
    }

    virtual ~TestCaseDenseMV_vDSP(){;}

    virtual void run() {

        if constexpr ( is_same< float,T >::value ) {
            vDSP_mmul( this->m_matrix, 1, this->m_vector, 1, this->m_output_vector, 1, this->m_M, 1, this->m_N );
        }
        else { // is_same< double,T >::value
            vDSP_mmulD( this->m_matrix, 1, this->m_vector, 1, this->m_output_vector, 1, this->m_M, 1, this->m_N );
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_NEON : public TestCaseDenseMV<T, IS_COL_MAJOR> {

  protected:
    const int m_factor_loop_unrolling;

  public:

    TestCaseDenseMV_NEON( const int M, const int N, const int factor_loop_unrolling )
        :TestCaseDenseMV<T, IS_COL_MAJOR>( M, N )
        ,m_factor_loop_unrolling(factor_loop_unrolling)
    {
        static_assert( is_same< float,T >::value || is_same< double,T >::value );

        this->setNEON( 1, factor_loop_unrolling );
    }

    virtual ~TestCaseDenseMV_NEON(){;}


    virtual void run() {

        calc_block( 0, this->m_M );
    }

    virtual void calc_block( const int row_begin, const int row_end_past_one ) {

        if constexpr (IS_COL_MAJOR) {

            switch( m_factor_loop_unrolling ) {

              case 1:
                run_col_major_loop_unrolling_1( row_begin, row_end_past_one );
                break;

              case 2:
                run_col_major_loop_unrolling_2( row_begin, row_end_past_one );
                break;

              case 4:
                run_col_major_loop_unrolling_4( row_begin, row_end_past_one );
                break;

              case 8:
              default:
                run_col_major_loop_unrolling_8( row_begin, row_end_past_one );
                break;
            }
        }
        else {
            switch( m_factor_loop_unrolling ) {

              case 1:
                run_row_major_loop_unrolling_1( row_begin, row_end_past_one );
                break;

              case 2:
                run_row_major_loop_unrolling_2( row_begin, row_end_past_one );
                break;

              case 4:
                run_row_major_loop_unrolling_4( row_begin, row_end_past_one );
                break;

              case 8:
              default:
                run_row_major_loop_unrolling_8( row_begin, row_end_past_one );
                break;
            }
        }
    }

    virtual void run_col_major_loop_unrolling_1( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i += 4 ) {

                float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const float col_v = this->m_vector[j];
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ]) );
                    const float32x4_t qw_col = { col_v, col_v, col_v, col_v };
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );

                    qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(float)*4);
            }
        }
        else {

            for ( int i = row_begin; i < row_end_past_one; i += 2 ) {

                float64x2_t qw_row_sum1 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const float col_v = this->m_vector[j];
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ]) );
                    const float64x2_t qw_col = { col_v, col_v };
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );

                    qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(double)*2);
            }
        }
    }

    virtual void run_col_major_loop_unrolling_2( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i += 8 ) {

                float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 4 ;
                    const float col_v = this->m_vector[j];
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ]) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ]) );
                    const float32x4_t qw_col = { col_v, col_v, col_v, col_v };
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );

                    qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+4]), &qw_row_sum2, sizeof(float)*4);
            }
        }
        else {

            for ( int i = row_begin; i < row_end_past_one; i += 4 ) {

                float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                float64x2_t qw_row_sum2 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 2 ;
                    const float col_v = this->m_vector[j];
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ]) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ]) );
                    const float64x2_t qw_col = { col_v, col_v };
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );

                    qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+2]), &qw_row_sum2, sizeof(double)*2);
            }
        }
    }

    virtual void run_col_major_loop_unrolling_4( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i += 16 ) {

                float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum4 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  4 ;
                    const int mat_index3 = mat_index1 +  8 ;
                    const int mat_index4 = mat_index1 + 12 ;
                    const float col_v = this->m_vector[j];
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ]) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ]) );
                    const float32x4_t qw_mat3 = vld1q_f32( &(this->m_matrix[ mat_index3 ]) );
                    const float32x4_t qw_mat4 = vld1q_f32( &(this->m_matrix[ mat_index4 ]) );
                    const float32x4_t qw_col = { col_v, col_v, col_v, col_v };
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );
                    const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col );
                    const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col );

                    qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                    qw_row_sum3 = vaddq_f32( qw_mc3, qw_row_sum3 );
                    qw_row_sum4 = vaddq_f32( qw_mc4, qw_row_sum4 );
                }

                memcpy(&(this->m_output_vector[i   ]), &qw_row_sum1, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+ 4]), &qw_row_sum2, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+ 8]), &qw_row_sum3, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+12]), &qw_row_sum4, sizeof(float)*4);
            }
        }
        else {

            for ( int i = row_begin; i < row_end_past_one; i += 8 ) {

                float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                float64x2_t qw_row_sum2 = { 0.0, 0.0 };
                float64x2_t qw_row_sum3 = { 0.0, 0.0 };
                float64x2_t qw_row_sum4 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 2 ;
                    const int mat_index3 = mat_index1 + 4 ;
                    const int mat_index4 = mat_index1 + 6 ;
                    const float col_v = this->m_vector[j];
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ]) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ]) );
                    const float64x2_t qw_mat3 = vld1q_f64( &(this->m_matrix[ mat_index3 ]) );
                    const float64x2_t qw_mat4 = vld1q_f64( &(this->m_matrix[ mat_index4 ]) );
                    const float64x2_t qw_col = { col_v, col_v };
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );
                    const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col );
                    const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col );

                    qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                    qw_row_sum3 = vaddq_f64( qw_mc3, qw_row_sum3 );
                    qw_row_sum4 = vaddq_f64( qw_mc4, qw_row_sum4 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+2]), &qw_row_sum2, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+4]), &qw_row_sum3, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+6]), &qw_row_sum4, sizeof(double)*2);
            }
        }
    }


    virtual void run_col_major_loop_unrolling_8( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i += 32 ) {

                float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum4 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum5 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum6 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum7 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum8 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  4 ;
                    const int mat_index3 = mat_index1 +  8 ;
                    const int mat_index4 = mat_index1 + 12 ;
                    const int mat_index5 = mat_index1 + 16 ;
                    const int mat_index6 = mat_index1 + 20 ;
                    const int mat_index7 = mat_index1 + 24 ;
                    const int mat_index8 = mat_index1 + 28 ;
                    const float col_v = this->m_vector[j];
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ]) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ]) );
                    const float32x4_t qw_mat3 = vld1q_f32( &(this->m_matrix[ mat_index3 ]) );
                    const float32x4_t qw_mat4 = vld1q_f32( &(this->m_matrix[ mat_index4 ]) );
                    const float32x4_t qw_mat5 = vld1q_f32( &(this->m_matrix[ mat_index5 ]) );
                    const float32x4_t qw_mat6 = vld1q_f32( &(this->m_matrix[ mat_index6 ]) );
                    const float32x4_t qw_mat7 = vld1q_f32( &(this->m_matrix[ mat_index7 ]) );
                    const float32x4_t qw_mat8 = vld1q_f32( &(this->m_matrix[ mat_index8 ]) );
                    const float32x4_t qw_col = { col_v, col_v, col_v, col_v };
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );
                    const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col );
                    const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col );
                    const float32x4_t qw_mc5  = vmulq_f32( qw_mat5, qw_col );
                    const float32x4_t qw_mc6  = vmulq_f32( qw_mat6, qw_col );
                    const float32x4_t qw_mc7  = vmulq_f32( qw_mat7, qw_col );
                    const float32x4_t qw_mc8  = vmulq_f32( qw_mat8, qw_col );

                    qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                    qw_row_sum3 = vaddq_f32( qw_mc3, qw_row_sum3 );
                    qw_row_sum4 = vaddq_f32( qw_mc4, qw_row_sum4 );
                    qw_row_sum5 = vaddq_f32( qw_mc5, qw_row_sum5 );
                    qw_row_sum6 = vaddq_f32( qw_mc6, qw_row_sum6 );
                    qw_row_sum7 = vaddq_f32( qw_mc7, qw_row_sum7 );
                    qw_row_sum8 = vaddq_f32( qw_mc8, qw_row_sum8 );
                }

                memcpy(&(this->m_output_vector[i   ]), &qw_row_sum1, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+ 4]), &qw_row_sum2, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+ 8]), &qw_row_sum3, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+12]), &qw_row_sum4, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+16]), &qw_row_sum5, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+20]), &qw_row_sum6, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+24]), &qw_row_sum7, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+28]), &qw_row_sum8, sizeof(float)*4);
            }
        }
        else {

            for ( int i = row_begin; i < row_end_past_one; i += 16 ) {

                float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                float64x2_t qw_row_sum2 = { 0.0, 0.0 };
                float64x2_t qw_row_sum3 = { 0.0, 0.0 };
                float64x2_t qw_row_sum4 = { 0.0, 0.0 };
                float64x2_t qw_row_sum5 = { 0.0, 0.0 };
                float64x2_t qw_row_sum6 = { 0.0, 0.0 };
                float64x2_t qw_row_sum7 = { 0.0, 0.0 };
                float64x2_t qw_row_sum8 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  2 ;
                    const int mat_index3 = mat_index1 +  4 ;
                    const int mat_index4 = mat_index1 +  6 ;
                    const int mat_index5 = mat_index1 +  8 ;
                    const int mat_index6 = mat_index1 + 10 ;
                    const int mat_index7 = mat_index1 + 12 ;
                    const int mat_index8 = mat_index1 + 14 ;
                    const float col_v = this->m_vector[j];
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ]) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ]) );
                    const float64x2_t qw_mat3 = vld1q_f64( &(this->m_matrix[ mat_index3 ]) );
                    const float64x2_t qw_mat4 = vld1q_f64( &(this->m_matrix[ mat_index4 ]) );
                    const float64x2_t qw_mat5 = vld1q_f64( &(this->m_matrix[ mat_index5 ]) );
                    const float64x2_t qw_mat6 = vld1q_f64( &(this->m_matrix[ mat_index6 ]) );
                    const float64x2_t qw_mat7 = vld1q_f64( &(this->m_matrix[ mat_index7 ]) );
                    const float64x2_t qw_mat8 = vld1q_f64( &(this->m_matrix[ mat_index8 ]) );
                    const float64x2_t qw_col = { col_v, col_v };
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );
                    const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col );
                    const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col );
                    const float64x2_t qw_mc5  = vmulq_f64( qw_mat5, qw_col );
                    const float64x2_t qw_mc6  = vmulq_f64( qw_mat6, qw_col );
                    const float64x2_t qw_mc7  = vmulq_f64( qw_mat7, qw_col );
                    const float64x2_t qw_mc8  = vmulq_f64( qw_mat8, qw_col );

                    qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                    qw_row_sum3 = vaddq_f64( qw_mc3, qw_row_sum3 );
                    qw_row_sum4 = vaddq_f64( qw_mc4, qw_row_sum4 );
                    qw_row_sum5 = vaddq_f64( qw_mc5, qw_row_sum5 );
                    qw_row_sum6 = vaddq_f64( qw_mc6, qw_row_sum6 );
                    qw_row_sum7 = vaddq_f64( qw_mc7, qw_row_sum7 );
                    qw_row_sum8 = vaddq_f64( qw_mc8, qw_row_sum8 );
                }

                memcpy(&(this->m_output_vector[i   ]), &qw_row_sum1, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+ 2]), &qw_row_sum2, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+ 4]), &qw_row_sum3, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+ 6]), &qw_row_sum4, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+ 8]), &qw_row_sum5, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+10]), &qw_row_sum6, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+12]), &qw_row_sum7, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+14]), &qw_row_sum8, sizeof(double)*2);
            }
        }
    }


    virtual void run_row_major_loop_unrolling_1( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=4 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ] ) );
                    const float32x4_t qw_col1 = vld1q_f32( &(this->m_vector[ j     ]      ) );
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );

                    qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                }
                this->m_output_vector[i] = qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3];
            }
        }
        else {
            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=2 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ] ) );
                    const float64x2_t qw_col1 = vld1q_f64( &(this->m_vector[ j     ]      ) );
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );

                    qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                }
                this->m_output_vector[i] = qw_lanewise_sum1[0] + qw_lanewise_sum1[1];
            }
        }
    }


    virtual void run_row_major_loop_unrolling_2( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=8 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 4;
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ] ) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ] ) );
                    const float32x4_t qw_col1 = vld1q_f32( &(this->m_vector[ j     ]      ) );
                    const float32x4_t qw_col2 = vld1q_f32( &(this->m_vector[ j + 4 ]      ) );
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );

                    qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );


                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                           + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3];
            }
        }
        else {
            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=4 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 2;
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ] ) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ] ) );
                    const float64x2_t qw_col1 = vld1q_f64( &(this->m_vector[ j     ]      ) );
                    const float64x2_t qw_col2 = vld1q_f64( &(this->m_vector[ j + 2 ]      ) );
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );

                    qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                }
                this->m_output_vector[i] = qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1];
            }
        }
    }

    virtual void run_row_major_loop_unrolling_4( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum4 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=16 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  4;
                    const int mat_index3 = mat_index1 +  8;
                    const int mat_index4 = mat_index1 + 12;
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ] ) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ] ) );
                    const float32x4_t qw_mat3 = vld1q_f32( &(this->m_matrix[ mat_index3 ] ) );
                    const float32x4_t qw_mat4 = vld1q_f32( &(this->m_matrix[ mat_index4 ] ) );
                    const float32x4_t qw_col1 = vld1q_f32( &(this->m_vector[ j      ]      ) );
                    const float32x4_t qw_col2 = vld1q_f32( &(this->m_vector[ j +  4 ]      ) );
                    const float32x4_t qw_col3 = vld1q_f32( &(this->m_vector[ j +  8 ]      ) );
                    const float32x4_t qw_col4 = vld1q_f32( &(this->m_vector[ j + 12 ]      ) );
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                    const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col3 );
                    const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col4 );

                    qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                    qw_lanewise_sum3 = vaddq_f32( qw_mc3, qw_lanewise_sum3 );
                    qw_lanewise_sum4 = vaddq_f32( qw_mc4, qw_lanewise_sum4 );
                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                           + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3]
                                           + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum3[2] + qw_lanewise_sum3[3]
                                           + qw_lanewise_sum4[0] + qw_lanewise_sum4[1] + qw_lanewise_sum4[2] + qw_lanewise_sum4[3];
            }
        }
        else {
            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum3 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum4 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=8 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 2;
                    const int mat_index3 = mat_index1 + 4;
                    const int mat_index4 = mat_index1 + 6;
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ] ) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ] ) );
                    const float64x2_t qw_mat3 = vld1q_f64( &(this->m_matrix[ mat_index3 ] ) );
                    const float64x2_t qw_mat4 = vld1q_f64( &(this->m_matrix[ mat_index4 ] ) );
                    const float64x2_t qw_col1 = vld1q_f64( &(this->m_vector[ j     ]      ) );
                    const float64x2_t qw_col2 = vld1q_f64( &(this->m_vector[ j + 2 ]      ) );
                    const float64x2_t qw_col3 = vld1q_f64( &(this->m_vector[ j + 4 ]      ) );
                    const float64x2_t qw_col4 = vld1q_f64( &(this->m_vector[ j + 6 ]      ) );
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                    const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col3 );
                    const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col4 );

                    qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                    qw_lanewise_sum3 = vaddq_f64( qw_mc3, qw_lanewise_sum3 );
                    qw_lanewise_sum4 = vaddq_f64( qw_mc4, qw_lanewise_sum4 );
                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1]
                                           + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum4[0] + qw_lanewise_sum4[1];
            }
        }
    }


    virtual void run_row_major_loop_unrolling_8( const int row_begin, const int row_end_past_one ) {

        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum4 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum5 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum6 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum7 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum8 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=32 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  4;
                    const int mat_index3 = mat_index1 +  8;
                    const int mat_index4 = mat_index1 + 12;
                    const int mat_index5 = mat_index1 + 16;
                    const int mat_index6 = mat_index1 + 20;
                    const int mat_index7 = mat_index1 + 24;
                    const int mat_index8 = mat_index1 + 28;
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ] ) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ] ) );
                    const float32x4_t qw_mat3 = vld1q_f32( &(this->m_matrix[ mat_index3 ] ) );
                    const float32x4_t qw_mat4 = vld1q_f32( &(this->m_matrix[ mat_index4 ] ) );
                    const float32x4_t qw_mat5 = vld1q_f32( &(this->m_matrix[ mat_index5 ] ) );
                    const float32x4_t qw_mat6 = vld1q_f32( &(this->m_matrix[ mat_index6 ] ) );
                    const float32x4_t qw_mat7 = vld1q_f32( &(this->m_matrix[ mat_index7 ] ) );
                    const float32x4_t qw_mat8 = vld1q_f32( &(this->m_matrix[ mat_index8 ] ) );
                    const float32x4_t qw_col1 = vld1q_f32( &(this->m_vector[ j      ]     ) );
                    const float32x4_t qw_col2 = vld1q_f32( &(this->m_vector[ j +  4 ]     ) );
                    const float32x4_t qw_col3 = vld1q_f32( &(this->m_vector[ j +  8 ]     ) );
                    const float32x4_t qw_col4 = vld1q_f32( &(this->m_vector[ j + 12 ]     ) );
                    const float32x4_t qw_col5 = vld1q_f32( &(this->m_vector[ j + 16 ]     ) );
                    const float32x4_t qw_col6 = vld1q_f32( &(this->m_vector[ j + 20 ]     ) );
                    const float32x4_t qw_col7 = vld1q_f32( &(this->m_vector[ j + 24 ]     ) );
                    const float32x4_t qw_col8 = vld1q_f32( &(this->m_vector[ j + 28 ]     ) );
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                    const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col3 );
                    const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col4 );
                    const float32x4_t qw_mc5  = vmulq_f32( qw_mat5, qw_col5 );
                    const float32x4_t qw_mc6  = vmulq_f32( qw_mat6, qw_col6 );
                    const float32x4_t qw_mc7  = vmulq_f32( qw_mat7, qw_col7 );
                    const float32x4_t qw_mc8  = vmulq_f32( qw_mat8, qw_col8 );

                    qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                    qw_lanewise_sum3 = vaddq_f32( qw_mc3, qw_lanewise_sum3 );
                    qw_lanewise_sum4 = vaddq_f32( qw_mc4, qw_lanewise_sum4 );
                    qw_lanewise_sum5 = vaddq_f32( qw_mc5, qw_lanewise_sum5 );
                    qw_lanewise_sum6 = vaddq_f32( qw_mc6, qw_lanewise_sum6 );
                    qw_lanewise_sum7 = vaddq_f32( qw_mc7, qw_lanewise_sum7 );
                    qw_lanewise_sum8 = vaddq_f32( qw_mc8, qw_lanewise_sum8 );
                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                           + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3]
                                           + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum3[2] + qw_lanewise_sum3[3]
                                           + qw_lanewise_sum4[0] + qw_lanewise_sum4[1] + qw_lanewise_sum4[2] + qw_lanewise_sum4[3]
                                           + qw_lanewise_sum5[0] + qw_lanewise_sum5[1] + qw_lanewise_sum5[2] + qw_lanewise_sum5[3]
                                           + qw_lanewise_sum6[0] + qw_lanewise_sum6[1] + qw_lanewise_sum6[2] + qw_lanewise_sum6[3]
                                           + qw_lanewise_sum7[0] + qw_lanewise_sum7[1] + qw_lanewise_sum7[2] + qw_lanewise_sum7[3]
                                           + qw_lanewise_sum8[0] + qw_lanewise_sum8[1] + qw_lanewise_sum8[2] + qw_lanewise_sum8[3];
            }
        }
        else {
            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum3 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum4 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum5 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum6 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum7 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum8 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=16 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  2;
                    const int mat_index3 = mat_index1 +  4;
                    const int mat_index4 = mat_index1 +  6;
                    const int mat_index5 = mat_index1 +  8;
                    const int mat_index6 = mat_index1 + 10;
                    const int mat_index7 = mat_index1 + 12;
                    const int mat_index8 = mat_index1 + 14;
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ] ) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ] ) );
                    const float64x2_t qw_mat3 = vld1q_f64( &(this->m_matrix[ mat_index3 ] ) );
                    const float64x2_t qw_mat4 = vld1q_f64( &(this->m_matrix[ mat_index4 ] ) );
                    const float64x2_t qw_mat5 = vld1q_f64( &(this->m_matrix[ mat_index5 ] ) );
                    const float64x2_t qw_mat6 = vld1q_f64( &(this->m_matrix[ mat_index6 ] ) );
                    const float64x2_t qw_mat7 = vld1q_f64( &(this->m_matrix[ mat_index7 ] ) );
                    const float64x2_t qw_mat8 = vld1q_f64( &(this->m_matrix[ mat_index8 ] ) );
                    const float64x2_t qw_col1 = vld1q_f64( &(this->m_vector[ j      ]     ) );
                    const float64x2_t qw_col2 = vld1q_f64( &(this->m_vector[ j +  2 ]     ) );
                    const float64x2_t qw_col3 = vld1q_f64( &(this->m_vector[ j +  4 ]     ) );
                    const float64x2_t qw_col4 = vld1q_f64( &(this->m_vector[ j +  6 ]     ) );
                    const float64x2_t qw_col5 = vld1q_f64( &(this->m_vector[ j +  8 ]     ) );
                    const float64x2_t qw_col6 = vld1q_f64( &(this->m_vector[ j + 10 ]     ) );
                    const float64x2_t qw_col7 = vld1q_f64( &(this->m_vector[ j + 12 ]     ) );
                    const float64x2_t qw_col8 = vld1q_f64( &(this->m_vector[ j + 14 ]     ) );
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                    const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col3 );
                    const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col4 );
                    const float64x2_t qw_mc5  = vmulq_f64( qw_mat5, qw_col5 );
                    const float64x2_t qw_mc6  = vmulq_f64( qw_mat6, qw_col6 );
                    const float64x2_t qw_mc7  = vmulq_f64( qw_mat7, qw_col7 );
                    const float64x2_t qw_mc8  = vmulq_f64( qw_mat8, qw_col8 );

                    qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                    qw_lanewise_sum3 = vaddq_f64( qw_mc3, qw_lanewise_sum3 );
                    qw_lanewise_sum4 = vaddq_f64( qw_mc4, qw_lanewise_sum4 );
                    qw_lanewise_sum5 = vaddq_f64( qw_mc5, qw_lanewise_sum5 );
                    qw_lanewise_sum6 = vaddq_f64( qw_mc6, qw_lanewise_sum6 );
                    qw_lanewise_sum7 = vaddq_f64( qw_mc7, qw_lanewise_sum7 );
                    qw_lanewise_sum8 = vaddq_f64( qw_mc8, qw_lanewise_sum8 );
                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1]
                                           + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum4[0] + qw_lanewise_sum4[1]
                                           + qw_lanewise_sum5[0] + qw_lanewise_sum5[1] + qw_lanewise_sum6[0] + qw_lanewise_sum6[1]
                                           + qw_lanewise_sum7[0] + qw_lanewise_sum7[1] + qw_lanewise_sum8[0] + qw_lanewise_sum8[1];
            }
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_NEON_multithread : public TestCaseDenseMV_NEON<T, IS_COL_MAJOR> {

    const size_t                m_num_threads;
    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    vector<thread>              m_threads;

  public:
    TestCaseDenseMV_NEON_multithread( const int M, const int N, const int factor_loop_unrolling, const int num_threads )
        :TestCaseDenseMV_NEON<T, IS_COL_MAJOR>( M, N, factor_loop_unrolling )
        ,m_num_threads( num_threads )
        ,m_fan_out    ( num_threads )
        ,m_fan_in     ( num_threads )
    {
        assert ( M % 4 == 0 && N % 4 == 0 && M >= 4 && N >= 4 );

        this->setNEON( num_threads, factor_loop_unrolling );

        const size_t num_rows_per_thread = this->m_M / m_num_threads;

        auto thread_lambda = [ this, num_rows_per_thread ]( const size_t thread_index ) {

            const size_t row_begin = thread_index * num_rows_per_thread;
            const size_t row_end   = row_begin + num_rows_per_thread;

            while ( true ) {

                m_fan_out.wait( thread_index );
                if( m_fan_out.isTerminating() ) {
                    break;
                }

                this->calc_block( row_begin, row_end );

                m_fan_in.notify();
                if( m_fan_in.isTerminating() ) {
                    break;
                }
            }
        };

        for ( size_t i = 0; i < num_threads; i++ ) {
            m_threads.emplace_back( thread_lambda, i );
        }
    }

    virtual ~TestCaseDenseMV_NEON_multithread() {

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


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_metal : public TestCaseDenseMV< T, IS_COL_MAJOR > {

    DenseMatrixVectorMetalCpp m_metal;

  public:

    TestCaseDenseMV_metal( const int M, const int N, const bool threads_over_rows )
        :TestCaseDenseMV<T, IS_COL_MAJOR>( M, N )
        ,m_metal( M, N , IS_COL_MAJOR, threads_over_rows )
    {
        this->setMetal( threads_over_rows ? THREADS_OVER_ROWS : THREADS_OVER_COLUMNS, 1, 1 );
    }

    virtual ~TestCaseDenseMV_metal(){;}

    virtual void setInitialStates( T* mat, T* vec, T* out_vec ) {

        m_metal.setInitialStates( mat, vec );

        TestCaseDenseMV<T, IS_COL_MAJOR>::setInitialStates( mat, vec, out_vec );
    }

    virtual void run() {

        m_metal.performComputation();
    }

    virtual T* getOutputVector() {
        return m_metal.getRawPointerOutVec();
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_MPS : public TestCaseDenseMV< T, IS_COL_MAJOR > {

    DenseMatrixVectorMetalCpp m_metal;

  public:

    TestCaseDenseMV_MPS( const int M, const int N )
        :TestCaseDenseMV<T, IS_COL_MAJOR>( M, N )
        ,m_metal( M, N )
    {
        static_assert( ! IS_COL_MAJOR );
        this->setMetal( MPS, 1, 1 );
    }

    virtual ~TestCaseDenseMV_MPS(){;}

    virtual void setInitialStates( T* mat, T* vec, T* out_vec ) {

        m_metal.setInitialStates( mat, vec );

        TestCaseDenseMV<T, IS_COL_MAJOR>::setInitialStates( mat, vec, out_vec );
    }

    virtual void run() {

        m_metal.performComputation();
    }

    virtual T* getOutputVector() {
        return m_metal.getRawPointerOutVec();
    }
};


template <class T, bool IS_COL_MAJOR>
class TestExecutorDenseMV : public TestExecutor {

  protected:

    const int             m_M;
    const int             m_N;
    default_random_engine m_e;
    T*                    m_matrix;
    T*                    m_vector;
    T*                    m_output_vector;
    T*                    m_output_vector_baseline;

  public:
    TestExecutorDenseMV(
        ostream&   os,
        const int  M,
        const int  N,
        const int  num_trials,
        const bool repeatable,
        const T    low,
        const T    high
    )
        :TestExecutor             ( os, num_trials )
        ,m_M                      ( M )
        ,m_N                      ( N )
        ,m_e                      ( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_matrix                 ( new T [ M * N ] )
        ,m_vector                 ( new T [ N ] )
        ,m_output_vector          ( new T [ M ] )
        ,m_output_vector_baseline ( new T [ M ] )
    {
        memset( m_output_vector,          0, sizeof(T)*m_M );
        memset( m_output_vector_baseline, 0, sizeof(T)*m_M );

        generateDenseMatrixVector( m_M, m_N, m_matrix, m_vector, low, high, m_e );
    }

    virtual ~TestExecutorDenseMV() {

        delete[] m_matrix;
        delete[] m_vector;
        delete[] m_output_vector;
        delete[] m_output_vector_baseline;
    }

    void prepareForRun ( const int test_case, const int num ) {

        memset( m_output_vector, 0,  sizeof(T) * m_M );

        auto t = dynamic_pointer_cast< TestCaseDenseMV<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_matrix, m_vector, m_output_vector );
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCaseDenseMV<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            memcpy( m_output_vector_baseline, m_output_vector, sizeof(T) * m_M );
        }
        t->compareTruth( m_output_vector_baseline );
    }

    const string preamble () {
        return "dims:[" + to_string(m_M) + ", " + to_string(m_N) +  "]";
    }
};


static const size_t NUM_TRIALS = 10;

struct matrix_dim {
    size_t M;
    size_t N;
};

struct matrix_dim matrix_dims[]={ {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}, {8*1024, 8*1024}, {16*1024, 16* 1024} };

template<class T, bool IS_COL_MAJOR>
void testSuitePerType ( const T gen_low, const T gen_high ) {

    const int neon_num_lanes = ( is_same<float, T>::value )? 4 : 2;

    for( auto& dims : matrix_dims ) {

	const auto M = dims.M;
	const auto N = dims.N;

        const auto span_neon = IS_COL_MAJOR ? M : N;

	TestExecutorDenseMV<T, IS_COL_MAJOR> e( cout, M, N, NUM_TRIALS, false, gen_low, gen_high );

        e.addTestCase( make_shared< TestCaseDenseMV_baseline    <T, IS_COL_MAJOR> > ( M, N ) );

        e.addTestCase( make_shared< TestCaseDenseMV_multithread <T, IS_COL_MAJOR> > ( M, N,  2 ) );
        e.addTestCase( make_shared< TestCaseDenseMV_multithread <T, IS_COL_MAJOR> > ( M, N,  4 ) );
        e.addTestCase( make_shared< TestCaseDenseMV_multithread <T, IS_COL_MAJOR> > ( M, N,  8 ) );

        e.addTestCase( make_shared< TestCaseDenseMV_NEON <T, IS_COL_MAJOR> > ( M, N,  1 ) );
        if ( span_neon >= 2 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON <T, IS_COL_MAJOR> > ( M, N,  2 ) );
        }
        if ( span_neon >= 4 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON <T, IS_COL_MAJOR> > ( M, N,  4 ) );
        }
        if ( span_neon >= 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON <T, IS_COL_MAJOR> > ( M, N,  8 ) );
        }
        if ( span_neon >= 2 * 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON_multithread <T, IS_COL_MAJOR> > ( M, N,  2, 8 ) );
        }
        if ( span_neon >= 4 * 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON_multithread <T, IS_COL_MAJOR> > ( M, N,  4, 8 ) );
        }
        if ( span_neon >= 8 * 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON_multithread <T, IS_COL_MAJOR> > ( M, N,  8, 8 ) );
        }

        e.addTestCase( make_shared< TestCaseDenseMV_blas <T, IS_COL_MAJOR> > ( M, N ) );

        if constexpr ( !IS_COL_MAJOR ) {
            e.addTestCase( make_shared< TestCaseDenseMV_vDSP <T, IS_COL_MAJOR> > ( M, N ) );
        }

        if constexpr ( !IS_COL_MAJOR && is_same< float,T >::value ) {
            e.addTestCase( make_shared< TestCaseDenseMV_MPS <T, IS_COL_MAJOR> > ( M, N ) );
        }  

        if constexpr ( is_same< float,T >::value ) {
            e.addTestCase( make_shared< TestCaseDenseMV_metal <T, IS_COL_MAJOR> > ( M, N, true  ) );
            e.addTestCase( make_shared< TestCaseDenseMV_metal <T, IS_COL_MAJOR> > ( M, N, false ) );
        }

	e.execute();
    }
}


int main( int argc, char* argv[] )
{
    TestCaseWithTimeMeasurements::printHeader( cout );

    cerr << "\n\nTesting for type float in column-major.\n\n";

    testSuitePerType<float, true > ( -1.0, 1.0 );

    cerr << "\n\nTesting for type float in row-major.\n\n";

    testSuitePerType<float, false > ( -1.0, 1.0 );

    cerr << "\n\nTesting for type double in column-major.\n\n";

    testSuitePerType<double, true > ( -1.0, 1.0 );

    cerr << "\n\nTesting for type double in row-major.\n\n";

    testSuitePerType<double, false > ( -1.0, 1.0 );

    return 0;
}
