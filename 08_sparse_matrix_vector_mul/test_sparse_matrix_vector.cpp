#include <Accelerate/Accelerate.h>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "thread_synchronizer.h"

#include "sparse_matrix_vector_metal_cpp.h"


template<class T>
class TestCaseSPMV :public TestCaseWithTimeMeasurements {

  protected:
    const int           m_M;
    const int           m_N;
    const int           m_num_nonzero_elems;
    int*                m_csr_row_ptrs;
    int*                m_csr_columns;
    T*                  m_csr_values;
    T*                  m_csr_vector;
    T*                  m_output_vector;

  public:

    TestCaseSPMV( const int M, const int N, const int num_nonzero_elems )
        :m_M                 ( M                 )
        ,m_N                 ( N                 )
        ,m_num_nonzero_elems ( num_nonzero_elems )
        ,m_csr_row_ptrs      ( nullptr           )
        ,m_csr_columns       ( nullptr           )
        ,m_csr_values        ( nullptr           )
        ,m_csr_vector        ( nullptr           )
        ,m_output_vector     ( nullptr           ) {

        static_assert( is_same< float,T >::value ||  is_same< double,T >::value );

        if constexpr ( is_same<float, T>::value ) {

            setDataElementType( FLOAT );
        }
        else if constexpr ( is_same<double, T>::value ) {

            setDataElementType( DOUBLE );
        }
        else {
            assert(true);
        }

        setMatrixSparse( M, N, num_nonzero_elems );

        setVerificationType( RMS );
    }

    virtual ~TestCaseSPMV(){;}

    virtual void compareTruth( const T* const baseline ) {
       
        auto rms = getRMSDiffTwoVectors( getOutputVector(), baseline, m_M );
        this->setRMS( rms );
    }

    virtual void setInitialStates( int* csr_row_ptrs, int* csr_columns, T* csr_values, T* csr_vector, T* output_vector ) {
        m_csr_row_ptrs  = csr_row_ptrs;
        m_csr_columns   = csr_columns;
        m_csr_values    = csr_values;
        m_csr_vector    = csr_vector;
        m_output_vector = output_vector;
    }

    virtual T* getOutputVector() {
        return m_output_vector;
    }

    virtual void run() = 0;
};


template<class T>
class TestCaseSPMV_baseline : public TestCaseSPMV<T> {

  public:
    TestCaseSPMV_baseline( const int M, const int N, const int num_nonzero_elems )
        :TestCaseSPMV<T>( M, N, num_nonzero_elems  )
    {
        this->setCPPBlock( 1, 1 );
    }

    virtual ~TestCaseSPMV_baseline(){;}

    virtual void run(){

        for (int i = 0; i < this->m_M; i++) {

            this->m_output_vector[i] = 0.0;

            for ( int j = this->m_csr_row_ptrs[i]; j < this->m_csr_row_ptrs[i+1]; j++ ){

                this->m_output_vector[i] += ( this->m_csr_values[j] * this->m_csr_vector[ this->m_csr_columns[j] ] ) ;
            }
        }
    }
};



template<class T>
class TestCaseSPMV_multithread : public TestCaseSPMV<T> {

    const size_t                m_num_threads;
    WaitNotifyMultipleWaiters   m_fan_out;
    WaitNotifyMultipleNotifiers m_fan_in;
    vector<thread>              m_threads;

  public:
    TestCaseSPMV_multithread( const int M, const int N, const int num_nonzero_elems, const size_t num_threads )
        :TestCaseSPMV<T>( M, N, num_nonzero_elems )
        ,m_num_threads ( num_threads )
        ,m_fan_out     ( num_threads )
        ,m_fan_in      ( num_threads )
    {
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

                    for ( int j = this->m_csr_row_ptrs[i]; j < this->m_csr_row_ptrs[i+1]; j++ ){

                        this->m_output_vector[i] += ( this->m_csr_values[j] * this->m_csr_vector[ this->m_csr_columns[j] ] ) ;
                    }
                }

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

    virtual ~TestCaseSPMV_multithread() {

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


template<class T>
class TestCaseSPMV_blas : public TestCaseSPMV<T> {

    sparse_matrix_float  m_mat_float;
    sparse_matrix_double m_mat_double;

  public:
    TestCaseSPMV_blas( const int M, const int N, const int num_nonzero_elems );
    virtual ~TestCaseSPMV_blas();

    virtual void run();

    virtual void setInitialStates( int* csr_row_ptrs, int* csr_columns, T* csr_values, T* csr_vector, T* output_vector );
};


template<>
TestCaseSPMV_blas<float>::TestCaseSPMV_blas( const int M, const int N, const int num_nonzero_elems )
    :TestCaseSPMV<float>( M, N, num_nonzero_elems )
    ,m_mat_float( nullptr ){
        setImplementationType(BLAS);
    }

template<>
TestCaseSPMV_blas<float>::~TestCaseSPMV_blas(){
    if ( m_mat_float != nullptr ) {
        sparse_matrix_destroy( m_mat_float );
    }
}

template<>
void TestCaseSPMV_blas<float>::run(){

    sparse_status s = sparse_matrix_vector_product_dense_float(
                          CblasNoTrans
                        , 1.0
                        , m_mat_float
                        , this->m_csr_vector
                        , 1
                        , this->m_output_vector
                        , 1 
                      );

    if ( s != SPARSE_SUCCESS ) {
           cerr << "sparse_matrix_vector_product_dense_float() failed.\n";
           return;
    }
}

template<>
void TestCaseSPMV_blas<float>::setInitialStates(
    int*   csr_row_ptrs,
    int*   csr_columns,
    float* csr_values,
    float* csr_vector,
    float* output_vector
) {
    TestCaseSPMV<float>::setInitialStates( csr_row_ptrs, csr_columns, csr_values, csr_vector, output_vector );

    if ( m_mat_float != nullptr ) {        
        sparse_matrix_destroy( m_mat_float );  
    }
    m_mat_float = sparse_matrix_create_float( this->m_M, this->m_N );

    for ( int i = 0; i < this->m_M; i++ ) {

        for ( int j = this->m_csr_row_ptrs[i]; j < this->m_csr_row_ptrs[i+1]; j++ ){

            sparse_status s = sparse_insert_entry_float( m_mat_float, this->m_csr_values[j], i, this->m_csr_columns[j] );

            if ( s != SPARSE_SUCCESS ) {

                cerr << "sparse_insert_entry_float() failed: " << i << ", " << j << "\n";
                return;
            }
        }
    }
         
    sparse_status s = sparse_commit( m_mat_float );
    if ( s != SPARSE_SUCCESS ) {
           cerr << "sparse_commit() failed.\n";
           return;
    }
}


template<>
TestCaseSPMV_blas<double>::TestCaseSPMV_blas( const int M, const int N, const int num_nonzero_elems )
    :TestCaseSPMV<double>( M, N, num_nonzero_elems )
    ,m_mat_double( nullptr ){
        setImplementationType(BLAS);
    }

template<>
TestCaseSPMV_blas<double>::~TestCaseSPMV_blas(){
    if ( m_mat_double != nullptr ) {
        sparse_matrix_destroy( m_mat_double );
    }
}

template<>
void TestCaseSPMV_blas<double>::run(){

    sparse_status s = sparse_matrix_vector_product_dense_double(
                          CblasNoTrans
                        , 1.0
                        , m_mat_double
                        , this->m_csr_vector
                        , 1
                        , this->m_output_vector
                        , 1 
                      );

    if ( s != SPARSE_SUCCESS ) {
           cerr << "sparse_matrix_vector_product_dense_double() failed.\n";
           return;
    }
}

template<>
void TestCaseSPMV_blas<double>::setInitialStates(
    int*    csr_row_ptrs,
    int*    csr_columns,
    double* csr_values,
    double* csr_vector,
    double* output_vector
) {
    TestCaseSPMV<double>::setInitialStates( csr_row_ptrs, csr_columns, csr_values, csr_vector, output_vector );

    if ( m_mat_double != nullptr ) {        
        sparse_matrix_destroy( m_mat_double );  
    }
    m_mat_double = sparse_matrix_create_double( this->m_M, this->m_N );

    for ( int i = 0; i < this->m_M; i++ ) {

        for ( int j = this->m_csr_row_ptrs[i]; j < this->m_csr_row_ptrs[i+1]; j++ ){

            sparse_status s = sparse_insert_entry_double( m_mat_double, this->m_csr_values[j], i, this->m_csr_columns[j] );

            if ( s != SPARSE_SUCCESS ) {

                cerr << "sparse_insert_entry_double() failed: " << i << ", " << j << "\n";
                return;
            }
        }
    }
         
    sparse_status s = sparse_commit( m_mat_double );
    if ( s != SPARSE_SUCCESS ) {
           cerr << "sparse_commit() failed.\n";
           return;
    }
}


template<class T>
class TestCaseSPMV_metal : public TestCaseSPMV<T> {

    SparseMatrixVectorMetalCpp m_metal;

    int*       m_csr_block_ptrs;
    int*       m_threads_per_row;
    int*       m_max_iters;
    int        m_num_blocks_plus_1;
    const bool m_use_adaptation;
    float      m_thread_utilization;

  public:

    TestCaseSPMV_metal( const int M, const int N, const int num_nonzero_elems, const bool use_adaptation )

        :TestCaseSPMV<T> ( M, N, num_nonzero_elems )
        ,m_metal  ( M, N, num_nonzero_elems, use_adaptation )
        ,m_csr_block_ptrs    ( nullptr )
        ,m_threads_per_row   ( nullptr )
        ,m_max_iters         ( nullptr )
        ,m_num_blocks_plus_1 ( 0 )
        ,m_use_adaptation    ( use_adaptation )
        ,m_thread_utilization(0.0)
    {
        this->setMetal( use_adaptation ? ADAPTIVE : NAIVE, 1, 1 );
    }

    virtual ~TestCaseSPMV_metal(){

        if ( m_csr_block_ptrs != nullptr ) {
            delete[] m_csr_block_ptrs;
        }        
        if ( m_threads_per_row != nullptr ) {
            delete[] m_threads_per_row;
        }        
        if ( m_max_iters != nullptr ) {
            delete[] m_max_iters;
        }        
    }

    virtual void run(){

        if ( m_use_adaptation ) {

            create_blocks();
        }

        m_metal.setInitialStates(
            m_csr_block_ptrs,
            m_threads_per_row,
            m_max_iters,
            m_num_blocks_plus_1 - 1,
            this->m_csr_row_ptrs,
            this->m_csr_columns,
            this->m_csr_values,
            this->m_csr_vector,
            this->m_output_vector
        );

        m_metal.performComputation();
    }

    virtual void compareTruth( const T* const baseline ) {

        memcpy( this->m_output_vector, m_metal.getRawPointerOutputVector(), sizeof(float) * (this->m_M) );
        return TestCaseSPMV<T>::compareTruth( baseline );
    }

    void create_blocks()
    {
        // this tries to pack consecutive rows to the current threadgroup
        // such that the thread utilization is maximalized in a greedy manner.

        vector<int> block_start_row;
        split_rows_into_threadgroups_greedy( block_start_row, 1024 );

        m_num_blocks_plus_1 = block_start_row.size();

        if ( m_csr_block_ptrs != nullptr ) {
            delete[] m_csr_block_ptrs;
        }        
        if ( m_threads_per_row != nullptr ) {
            delete[] m_threads_per_row;
        }        
        if ( m_max_iters != nullptr ) {
            delete[] m_max_iters;
        }        
        m_csr_block_ptrs  = new int[ m_num_blocks_plus_1 ];
        m_threads_per_row = new int[ m_num_blocks_plus_1 ];
        m_max_iters       = new int[ m_num_blocks_plus_1 ];

        check_split_blocks( block_start_row );
    }

    void split_rows_into_threadgroups_greedy(

        vector<int>& block_start_row,
        const int    num_threads_per_threadgroup
        
    ) {
        block_start_row.clear();

        int  sum_nnz_in_cur_block             = 0;
        int  start_row_in_cur_block           = 0;
        bool end_current_block_with_i_minus_1 = false;
        bool end_current_block_with_i         = false;

        for ( int i = 0; i < this->m_M; i++ ) {

            // assign one thread for empty row to avoid complication.
            auto num_nnz_on_row_i = max( 1, this->m_csr_row_ptrs[i+1] - this->m_csr_row_ptrs[i] );
        
            end_current_block_with_i_minus_1 = false;
            end_current_block_with_i         = false;

            if ( sum_nnz_in_cur_block + num_nnz_on_row_i > num_threads_per_threadgroup ) {

                if ( sum_nnz_in_cur_block < num_threads_per_threadgroup / 2 ) {

                    end_current_block_with_i = true;
                }
                else if (    sum_nnz_in_cur_block < (num_threads_per_threadgroup / 4) * 3 
                          && num_nnz_on_row_i < num_threads_per_threadgroup ) {

                    end_current_block_with_i = true;
                }      
                else {
                    end_current_block_with_i_minus_1 = true;
                }
            }
            if ( end_current_block_with_i_minus_1 ) {

                block_start_row.push_back( start_row_in_cur_block );
                start_row_in_cur_block = i;
                sum_nnz_in_cur_block   = num_nnz_on_row_i;
            }
            else if ( end_current_block_with_i ) {

                block_start_row.push_back( start_row_in_cur_block );
                start_row_in_cur_block = i + 1;
                sum_nnz_in_cur_block   = 0;
            }
            else {
                sum_nnz_in_cur_block += num_nnz_on_row_i;
            }
        }
        if ( block_start_row.size() == 0 ){
            block_start_row.push_back(0);
            block_start_row.push_back(this->m_M);
        }
        else if ( *block_start_row.rbegin() != this->m_M ){
            block_start_row.push_back( this->m_M );
        }
    }

    void check_split_blocks( const vector<int>& block_start_row ) {

        int sum_nnz_to_check = 0;

        for ( int b = 0 ; b < block_start_row.size() - 1; b++ ) {

            const int row_begin = block_start_row[ b     ];
            const int row_end   = block_start_row[ b + 1 ];
            const int num_rows  = row_end - row_begin;

            int min_col  = 0;
            int max_col  = 0;
            int sum_cols = 0;

            for ( int r = row_begin; r < row_end; r++ ) {

                const int col_begin = this->m_csr_row_ptrs[r];
                const int col_end   = this->m_csr_row_ptrs[r+1];
                const int num_cols  = col_end - col_begin;

                sum_cols += num_cols;

                if ( r == row_begin ) {

                    min_col = num_cols;
                    max_col = num_cols;
                }
                else {
                    min_col = (min_col > num_cols) ? num_cols : min_col;
                    max_col = (max_col < num_cols) ? num_cols : max_col;
                }
            }

            int num_sub_blocks;
            for ( num_sub_blocks = 1 ; num_sub_blocks < 1024; num_sub_blocks *=2 ) {
                if ( num_sub_blocks >= num_rows ) {
                    break;
                }
            }

            sum_nnz_to_check += sum_cols;

            int num_threads_per_sub_block = 1024 / num_sub_blocks;
            int max_iter_per_subblock     = ( max_col + num_threads_per_sub_block - 1 ) / num_threads_per_sub_block;

            m_thread_utilization =   (max_iter_per_subblock>0)
                                   ? ((float)(sum_cols)) / ((float)(1024 * max_iter_per_subblock))
                                   : 0.0
                                   ;

            m_csr_block_ptrs  [b] = row_begin;
            m_threads_per_row [b] = num_threads_per_sub_block;
            m_max_iters       [b] = max_iter_per_subblock;
        }

        m_csr_block_ptrs[ block_start_row.size() - 1 ] = (*block_start_row.rbegin());
    }
};


template <class T>
class TestExecutorSPMV : public TestExecutor {

  protected:

    const int             m_M;
    const int             m_N;
    const int             m_num_nonzero_elems;
    default_random_engine m_e;
    int*                  m_csr_row_ptrs;
    int*                  m_csr_columns;
    T*                    m_csr_values;
    T*                    m_csr_vector;
    T*                    m_output_vector;
    T*                    m_output_vector_baseline;

  public:

    TestExecutorSPMV(
        ostream&   os,
        const int  M,
        const int  N,
        const int  num_nonzero_elems,
        const int  num_trials,
        const bool repeatable,
        const T    low,
        const T    high
    )
        :TestExecutor( os, num_trials )
        ,m_M                      ( M )
        ,m_N                      ( N )
        ,m_num_nonzero_elems      ( num_nonzero_elems )
        ,m_e                      ( repeatable? 0 : chrono::system_clock::now().time_since_epoch().count() )
        ,m_csr_row_ptrs           ( new int [ M + 1 ] )
        ,m_csr_columns            ( new int [ num_nonzero_elems ] )
        ,m_csr_values             ( new T   [ num_nonzero_elems ] )
        ,m_csr_vector             ( new T   [ N ] )
        ,m_output_vector          ( new T   [ M ] )
        ,m_output_vector_baseline ( new T   [ M ] )
    {
        memset( m_output_vector,          0, sizeof(T)*m_M );
        memset( m_output_vector_baseline, 0, sizeof(T)*m_M );

        generateCSR( m_M, m_N, m_num_nonzero_elems, low, high, m_e, m_csr_row_ptrs, m_csr_columns, m_csr_values, m_csr_vector) ;
    }

    virtual ~TestExecutorSPMV() {

        delete[] m_csr_row_ptrs;
        delete[] m_csr_columns;
        delete[] m_csr_values;
        delete[] m_csr_vector;
        delete[] m_output_vector;
        delete[] m_output_vector_baseline;
    }

    void prepareForRun ( const int test_case, const int num ) {

        memset( m_output_vector, 0,  sizeof(T) * m_M );

        auto t = dynamic_pointer_cast< TestCaseSPMV<T> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_csr_row_ptrs, m_csr_columns, m_csr_values, m_csr_vector, m_output_vector );
    }

    void cleanupAfterBatchRuns ( const int test_case ) {
        auto t = dynamic_pointer_cast< TestCaseSPMV<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            memcpy( m_output_vector_baseline, m_output_vector, sizeof(T) * m_M );
        }
        t->compareTruth( m_output_vector_baseline );
    }
};

static const size_t NUM_TRIALS = 10;

struct matrix_dim {
    size_t M;
    size_t N;
    float  ratio;
};

struct matrix_dim matrix_dims[]={

      {     256,     256, 0.1  }
    , {     512,     512, 0.1  }
    , {    1024,    1024, 0.1  }
    , {    2048,    2048, 0.1  }
    , {    4096,    4096, 0.1  }
    , {  8*1024,  8*1024, 0.1  }
    , { 16*1024, 16*1024, 0.1  }  
//    , { 32*1024, 32*1024, 0.1  }  
};

template<class T>
void testSuitePerType ( const T gen_low, const T gen_high ) {

    for( auto& dims : matrix_dims ) {

        const auto M = dims.M;
        const auto N = dims.N;

        const int num_nonzero_elems = (int)( ((float)(M*N))*dims.ratio );

        TestExecutorSPMV<T> e( cout, M, N, num_nonzero_elems, NUM_TRIALS, false, gen_low, gen_high );

        e.addTestCase( make_shared< TestCaseSPMV_baseline    <T> > ( M, N, num_nonzero_elems        ) );
        e.addTestCase( make_shared< TestCaseSPMV_multithread <T> > ( M, N, num_nonzero_elems,  2    ) );
        e.addTestCase( make_shared< TestCaseSPMV_multithread <T> > ( M, N, num_nonzero_elems,  4    ) );
        e.addTestCase( make_shared< TestCaseSPMV_multithread <T> > ( M, N, num_nonzero_elems,  8    ) );
        e.addTestCase( make_shared< TestCaseSPMV_multithread <T> > ( M, N, num_nonzero_elems, 16    ) );
        e.addTestCase( make_shared< TestCaseSPMV_blas        <T> > ( M, N, num_nonzero_elems        ) );
        if constexpr ( is_same< float,T >::value ) {
            e.addTestCase( make_shared< TestCaseSPMV_metal       <T> > ( M, N, num_nonzero_elems, false ) );
            e.addTestCase( make_shared< TestCaseSPMV_metal       <T> > ( M, N, num_nonzero_elems, true  ) );
        }
        e.execute();
    }
}

int main( int argc, char* argv[] )
{
    TestCaseWithTimeMeasurements::printHeader( cout );

    cerr << "\n\nTesting for type float.\n\n";

    testSuitePerType<float> ( -1.0, 1.0 );

    cerr << "\n\nTesting for type double.\n\n";

    testSuitePerType<double> ( -1.0, 1.0 );

    return 0;
}
