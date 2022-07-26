#ifndef __TEST_CASE_TIME_MEASUREMENTS_H__
#define __TEST_CASE_TIME_MEASUREMENTS_H__

#include <type_traits>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <thread>
#include <vector>
#include <map>
#include <math.h>
#include <assert.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std;

/* Sorting out outputs
- Data type: float/double/int etc.
- Sub data type: column-major, row-major, vector
- Size: [X,Y, nnz] for vector, Y = 1
- ExecutionConfiguration
  - CPU Num Threads
  - Metal Num Groups/Grid
  - Metal Num Threads/Group
- Mean Time
- Stddev Time
- Correctness
  - Correct/Incorrect
  - RMS
  - Distance
*/


enum TestDataElementType { FLOAT, DOUBLE, INT };

inline ostream& printTestDataElementType( ostream& os, const TestDataElementType t ) {
    switch (t) {
      case FLOAT:
        os << "FLOAT";
        break;
      case DOUBLE:
        os << "DOUBLE";
        break;
      case INT:
        os << "INT";
        break;
      default:
        break;
    }

    return os;
}

enum TestDataElementSubtype {
    VECTOR,
    MATRIX_COL_MAJOR,
    MATRIX_ROW_MAJOR,
    MATRIX_SPARSE,
    STRUCTURE_OF_ARRAYS,
    ARRAY_OF_STRUCTURES,
    RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC,
    RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC,
    REAL_NONSYMMETRIC_MU02,
    REAL_NONSYMMETRIC_MU08,
    REAL_SYMMETRIC
};

inline ostream& printTestDataElementSubtype( ostream& os, const TestDataElementSubtype t ) {
    switch (t) {

      case VECTOR:
        os << "VECTOR";
        break;

      case MATRIX_COL_MAJOR:
        os << "MATRIX_COL_MAJOR";
        break;

      case MATRIX_ROW_MAJOR:
        os << "MATRIX_ROW_MAJOR";
        break;

      case MATRIX_SPARSE:
        os << "MATRIX_SPARSE";
        break;

      case STRUCTURE_OF_ARRAYS:
        os << "STRUCTURE_OF_ARRAYS";
        break;

      case  ARRAY_OF_STRUCTURES:
        os << "ARRAY_OF_STRUCTURES";
        break;

      case RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC:
        os << "RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC";
        break;

      case RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC:
        os << "RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC";
        break;

      case REAL_NONSYMMETRIC_MU02:
        os << "REAL_NONSYMMETRIC_MU02";
        break;

      case REAL_NONSYMMETRIC_MU08:
        os << "REAL_NONSYMMETRIC_MU08";
        break;

      case REAL_SYMMETRIC:
        os << "REAL_SYMMETRIC";
        break;

      default:
        break;
    }
    return os;
}

enum TestDataVerificationType { TRUE_FALSE, RMS, DISTANCE };

inline ostream& printTestDataVerificationType( ostream& os, const TestDataVerificationType t ) {

    switch (t) {

      case TRUE_FALSE:
        os << "TRUE_FALSE";
        break;

      case RMS:
        os << "RMS";
        break;

      case DISTANCE:
        os << "DISTANCE";
        break;

      default:
        break;
    }
    return os;
}

enum ExecConfigInterpretationForCharts { REPORT_ALL, REPORT_BEST };

inline ostream& printExecConfigInterpretationForCharts( ostream& os, const ExecConfigInterpretationForCharts t ) {

    switch (t) {

      case REPORT_ALL:
        os << "REPORT_ALL";
        break;

      case REPORT_BEST:
        os << "REPORT_BEST";
        break;

      default:
        break;
    }
    return os;
}

enum ImplementationType { CPP_BLOCK, CPP_INTERLEAVED, CPP_STDLIB, NEON, MEMCPY, VDSP, BLAS, VDSP_BLAS, BOOST_SPREAD_SORT, BOOST_SAMPLE_SORT, CIIMAGE_CPU, CIIMAGE_GPU, EIGEN3, GSL, LAPACK, LAPACK_WITH_MAT_INVERSE, METAL, CPP_COLUMN_CHOLESKY, CPP_SUBMATRIX_CHOLESKY, ACCELERATE, LEMKE_CPP, LEMKE_NEON, LEMKE_VDSP, LEMKE_BULLET_ORIGINAL, LEMKE_BULLET_IMPROVED_LEXICO_MINIMUM, FIXED_POINT_VDSP, PGS_VDSP, PGS_SM_VDSP };

inline ostream& printImplementationType( ostream& os, const  ImplementationType t ) {

    switch (t) {
      case CPP_BLOCK:
        os << "CPP_BLOCK";
        break;

      case CPP_INTERLEAVED:
        os << "CPP_INTERLEAVED";
        break;

      case CPP_STDLIB:
        os << "CPP_STDLIB";
        break;

      case MEMCPY:
        os << "MEMCPY";
        break;

      case NEON:
        os << "NEON";
        break;

      case VDSP:
        os << "VDSP";
        break;

      case BLAS:
        os << "BLAS";
        break;

      case VDSP_BLAS:
        os << "VDSP_BLAS";
        break;

      case BOOST_SPREAD_SORT:
        os << "BOOST_SPREAD_SORT";
        break;

      case BOOST_SAMPLE_SORT:
        os << "BOOST_SAMPLE_SORT";
        break;

      case CIIMAGE_CPU:
        os << "CIIMAGE_CPU";
        break;

      case CIIMAGE_GPU:
        os << "CIIMAGE_GPU";
        break;

      case EIGEN3:
        os << "EIGEN3";
        break;

      case GSL:
        os << "GSL";
        break;

      case LAPACK:
        os << "LAPACK";
        break;

      case LAPACK_WITH_MAT_INVERSE:
        os << "LAPACK_WITH_MAT_INVERSE";
        break;

      case METAL:
        os << "METAL";
        break;

      case CPP_COLUMN_CHOLESKY:
        os << "CPP_COLUMN_CHOLESKY";
        break;

      case CPP_SUBMATRIX_CHOLESKY:
        os << "CPP_SUBMATRIX_CHOLESKY";
        break;

      case ACCELERATE:
        os << "ACCELERATE";
        break;

      case LEMKE_CPP:
        os << "LEMKE_CPP";
        break;

      case LEMKE_NEON:
        os << "LEMKE_NEON";
        break;

      case LEMKE_VDSP:
        os << "LEMKE_VDSP";
        break;

      case LEMKE_BULLET_ORIGINAL:
        os << "LEMKE_BULLET_ORIGINAL";
        break;

      case LEMKE_BULLET_IMPROVED_LEXICO_MINIMUM:
        os << "LEMKE_BULLET_IMPROVED_LEXICO_MINIMUM";
        break;
            
      case FIXED_POINT_VDSP:
        os << "FIXED_POINT_VDSP";
        break;

      case PGS_VDSP:
        os << "PGS_VDSP";
        break;

      case PGS_SM_VDSP:
        os << "PGS_SM_VDSP";
        break;

      default:
        break;
    }
    return os;
}

enum MetalImplementationType {
    NOT_APPLICABLE,
    DEFAULT,
    DEFAULT_SHARED,
    DEFAULT_MANAGED,
    BLIT_SHARED,
    BLIT_MANAGED,
    TWO_PASS_DEVICE_MEMORY,
    TWO_PASS_SHARED_MEMORY,
    TWO_PASS_SIMD_SHUFFLE,
    TWO_PASS_SIMD_SUM,
    ONE_PASS_ATOMIC_SIMD_SHUFFLE,
    ONE_PASS_ATOMIC_SIMD_SUM,
    ONE_PASS_THREAD_COUNTER,
    SCAN_THEN_FAN,
    REDUCE_THEN_SCAN,
    MERRILL_AND_GRIMSHAW,
    COALESCED_WRITE,
    UNCOALESCED_WRITE,
    COALESCED_WRITE_EARLY_OUT,
    UNCOALESCED_WRITE_EARLY_OUT,
    COALESCED_WRITE_IN_ONE_COMMIT,
    UNCOALESCED_WRITE_IN_ONE_COMMIT,
    BITONIC_SORT,
    THREADS_OVER_ROWS,
    THREADS_OVER_COLUMNS,
    NAIVE,
    TWO_STAGES,
    MPS,
    ADAPTIVE,
    ONE_COMMIT,
    MULTIPLE_COMMITS
};

inline ostream& printMetalImplementationType( ostream& os, const  MetalImplementationType t ) {

    switch (t) {
      case NOT_APPLICABLE:
        os << "NOT_APPLICABLE";
        break;

      case DEFAULT:
        os << "DEFAULT";
        break;

      case DEFAULT_SHARED:
        os << "DEFAULT_SHARED";
        break;

      case DEFAULT_MANAGED:
        os << "DEFAULT_MANAGED";
        break;

      case BLIT_SHARED:
        os << "BLIT_SHARED";
        break;

      case BLIT_MANAGED:
        os << "BLIT_MANAGED";
        break;

      case TWO_PASS_DEVICE_MEMORY:
        os << "TWO_PASS_DEVICE_MEMORY";
        break;

      case TWO_PASS_SHARED_MEMORY:
        os << "TWO_PASS_SHARED_MEMORY";
        break;

      case TWO_PASS_SIMD_SHUFFLE:
        os << "TWO_PASS_SIMD_SHUFFLE";
        break;

      case TWO_PASS_SIMD_SUM:
        os << "TWO_PASS_SIMD_SUM";
        break;

      case ONE_PASS_ATOMIC_SIMD_SHUFFLE:
        os << "ONE_PASS_ATOMIC_SIMD_SHUFFLE";
        break;

      case ONE_PASS_ATOMIC_SIMD_SUM:
        os << "ONE_PASS_ATOMIC_SIMD_SUM";
        break;

      case ONE_PASS_THREAD_COUNTER:
        os << "ONE_PASS_THREAD_COUNTER";
        break;

      case SCAN_THEN_FAN:
        os << "SCAN_THEN_FAN";
        break;

      case REDUCE_THEN_SCAN:
        os << "REDUCE_THEN_SCAN";
        break;

      case MERRILL_AND_GRIMSHAW:
        os << "MERRILL_AND_GRIMSHAW";
        break;

      case COALESCED_WRITE:
        os << "COALESCED_WRITE";
        break;

      case UNCOALESCED_WRITE:
        os << "UNCOALESCED_WRITE";
        break;

      case COALESCED_WRITE_EARLY_OUT:
        os << "COALESCED_WRITE_EARLY_OUT";
        break;

      case UNCOALESCED_WRITE_EARLY_OUT:
        os << "UNCOALESCED_WRITE_EARLY_OUT";
        break;

      case COALESCED_WRITE_IN_ONE_COMMIT:
        os << "COALESCED_WRITE_IN_ONE_COMMIT";
        break;

      case UNCOALESCED_WRITE_IN_ONE_COMMIT:
        os << "UNCOALESCED_WRITE_IN_ONE_COMMIT";
        break;

      case BITONIC_SORT:
        os << "BITONIC_SORT";
        break;

      case THREADS_OVER_ROWS:
        os << "THREADS_OVER_ROWS";
        break;

      case THREADS_OVER_COLUMNS:
        os << "THREADS_OVER_COLUMNS";
        break;

      case NAIVE:
        os << "NAIVE";
        break;

      case TWO_STAGES:
        os << "TWO_STAGES";
        break;

      case MPS:
        os << "MPS";
        break;

      case ADAPTIVE:
        os << "ADAPTIVE";
        break;

      case ONE_COMMIT:
        os << "ONE_COMMIT";
        break;

      case MULTIPLE_COMMITS:
        os << "MULTIPLE_COMMITS";
        break;

      default:
        break;
    }
    return os;
}

class TestDataDimension {

  public:
    int m_row;
    int m_col;
    int m_nnz;

    TestDataDimension()
        :m_row(0)
        ,m_col(0)
        ,m_nnz(0)
    {;}

    void setVectorSize(const int dimension) {
        m_row = dimension;
        m_col = 0;
        m_nnz = 0;
    }

    void setMatrixDimension( const int row, const int col ) {
        m_row = row;
        m_col = col;
        m_nnz = 0;
    }

    void setMatrixDimension( const int row, const int col, const int nnz ) {
        m_row = row;
        m_col = col;
        m_nnz = nnz;
    }

    static void printHeader( ostream& os ) {
        os << "vector length/matrix row";
        os << "\t";
        os << "matrix columns";
        os << "\t";
        os << "number of non zeros";
    }

    ostream& print( ostream& os ) {
        os << m_row << "\t" <<  m_col << "\t" << m_nnz;
        return os;
    }

    virtual ~TestDataDimension() {;}
};

class ExecutionConfiguration {

  public:

    int m_factor_loop_unrolling;
    int m_num_threads_cpu;
    int m_num_groups_per_grid;
    int m_num_threads_per_group;

    ExecutionConfiguration()
        :m_factor_loop_unrolling ( 1 )
        ,m_num_threads_cpu       ( 1 )
        ,m_num_groups_per_grid   ( 0 )
        ,m_num_threads_per_group ( 0 )
    {;}

    void setNumThreads(const int n )
    {
        m_num_threads_cpu = n;
    }

    void setMetalConfiguration(const int groups_per_grid, const int threads_per_group )
    {
        m_num_groups_per_grid   = groups_per_grid;
        m_num_threads_per_group = threads_per_group;
    }

    void setNumThreadsCPU( const int n ) {
        m_num_threads_cpu = n;
    }

    void setFactorLoopUnrolling(const int d )
    {
        m_factor_loop_unrolling = d;
    }

    static void printHeader( ostream& os ) {
        os << "loop unrolling factor";
        os << "\t";
        os << "num CPU threads";
        os << "\t";
        os << "num groups per grid";
        os << "\t";
        os << "num threads per group";
    }

    ostream& print( ostream& os ) {
        os <<  m_factor_loop_unrolling << "\t" << m_num_threads_cpu << "\t" << m_num_groups_per_grid << "\t" << m_num_threads_per_group;
        return os;
    }

    virtual ~ExecutionConfiguration(){;}
};

class TestCaseConfiguration {

    TestDataElementType               m_data_type;
    TestDataElementSubtype            m_data_subtype;
    TestDataDimension                 m_dimension;
    ImplementationType                m_implementation_type;
    TestDataVerificationType          m_verification_type;
    ExecConfigInterpretationForCharts m_metal_report_type;
    MetalImplementationType           m_metal_implementation_type;
    ExecutionConfiguration            m_execution_configuration;

  public:
    TestCaseConfiguration()
        :m_data_type                 ( FLOAT          )
        ,m_data_subtype              ( VECTOR         )
        ,m_implementation_type       ( CPP_BLOCK      )
        ,m_verification_type         ( TRUE_FALSE     )
        ,m_metal_report_type         ( REPORT_BEST    )
        ,m_metal_implementation_type ( NOT_APPLICABLE )
    {;}

    void setDataElementType( const TestDataElementType data_type ) {
        m_data_type  = data_type;;
    }

    void setVector( const int dimension ) {
        m_data_subtype = VECTOR;
        m_dimension.setVectorSize( dimension );
    }

    void setMatrixColMajor( const int row, const int col ) {
        m_data_subtype = MATRIX_COL_MAJOR;
        m_dimension.setMatrixDimension( row, col );
    }

    void setMatrixRowMajor( const int row, const int col ) {
        m_data_subtype = MATRIX_ROW_MAJOR;
        m_dimension.setMatrixDimension( row, col );
    }

    void setMatrixSparse( const int row, const int col, const int nnz ) {
        m_data_subtype = MATRIX_SPARSE;
        m_dimension.setMatrixDimension( row, col, nnz );
    }

    void setSOA( const int dimension ) {
        m_data_subtype = STRUCTURE_OF_ARRAYS;
        m_dimension.setVectorSize( dimension );
    }

    void setAOS( const int dimension ) {
        m_data_subtype = ARRAY_OF_STRUCTURES;
        m_dimension.setVectorSize( dimension );
    }

    void setRandomDiagonallyDominantSkewSymmetric( const int dimension ) {
        m_data_subtype = RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC;
        m_dimension.setMatrixDimension( dimension, dimension );
    }

    void setRandomDiagonallyDominantSymmetric( const int dimension ) {
        m_data_subtype = RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC;
        m_dimension.setMatrixDimension( dimension, dimension );
    }

    void setRealNonsymmetricMu02( const int dimension ) {
        m_data_subtype = REAL_NONSYMMETRIC_MU02;
        m_dimension.setMatrixDimension( dimension, dimension );
    }

    void setRealNonsymmetricMu08( const int dimension ) {
        m_data_subtype = REAL_NONSYMMETRIC_MU08;
        m_dimension.setMatrixDimension( dimension, dimension );
    }

    void setRealSymmetric( const int dimension ) {
        m_data_subtype = REAL_SYMMETRIC;
        m_dimension.setMatrixDimension( dimension, dimension );
    }

    void setVerificationType( const TestDataVerificationType t ) {
        m_verification_type = t;
    }

    TestDataVerificationType verificationType() {
        return m_verification_type;
    }

    void setMemcpy( const int num_threads ) {
        m_implementation_type = MEMCPY;
        m_execution_configuration.setNumThreadsCPU( num_threads );
    }

    void setCPPBlock( const int num_threads, const int factor_loop_unrolling ) {
        m_implementation_type = CPP_BLOCK;
        m_execution_configuration.setNumThreadsCPU( num_threads );
        m_execution_configuration.setFactorLoopUnrolling( factor_loop_unrolling );
    }

    void setCPPInterleaved( const int num_threads, const int factor_loop_unrolling ) {
        m_implementation_type = CPP_INTERLEAVED;
        m_execution_configuration.setNumThreadsCPU( num_threads );
        m_execution_configuration.setFactorLoopUnrolling( factor_loop_unrolling );
    }

    void setNEON( const int num_threads, const int factor_loop_unrolling ) {
        m_implementation_type = NEON;
        m_execution_configuration.setNumThreadsCPU( num_threads );
        m_execution_configuration.setFactorLoopUnrolling( factor_loop_unrolling );
    }

    void setBoostSampleSort( const int num_threads ) {
        m_implementation_type = BOOST_SAMPLE_SORT;
        m_execution_configuration.setNumThreadsCPU( num_threads );
    }        

    void setMetal( const MetalImplementationType t, const int groups_per_grid, const int threads_per_group ) {
        m_implementation_type       = METAL;
        m_metal_implementation_type = t;
        m_execution_configuration.setMetalConfiguration( groups_per_grid, threads_per_group );
    }        

    void setImplementationType( const ImplementationType t ) {
        m_implementation_type  = t;
    }

    void setColumnCholesky() {
        m_implementation_type = CPP_COLUMN_CHOLESKY;
    }

    void setSubmatrixCholesky() {
        m_implementation_type = CPP_SUBMATRIX_CHOLESKY;
    }


    static void printHeader( ostream& os ) {

        os << "data element type";
        os << "\t";
        os << "data element subtype";
        os << "\t";
        TestDataDimension::printHeader(os);
        os << "\t";
        os << "implementation type";
        os << "\t";
        os << "metal implementation type";
        os << "\t";
        os << "configuration interpretation for charts";
        os << "\t";
        ExecutionConfiguration::printHeader(os);
        os << "\t";
        os << "test data verification type";
    }

    ostream& print( ostream& os ) {

        printTestDataElementType( os, m_data_type );
        os << "\t";
        printTestDataElementSubtype( os, m_data_subtype );
        os << "\t";
        m_dimension.print( os );
        os << "\t";
        printImplementationType( os, m_implementation_type );
        os << "\t";
        printMetalImplementationType( os, m_metal_implementation_type );
        os << "\t";
        printExecConfigInterpretationForCharts( os, m_metal_report_type );
        os << "\t";
        m_execution_configuration.print( os );
        os << "\t";
        printTestDataVerificationType( os, m_verification_type );
        return os;
    }

    virtual ostream& printBrief( ostream& os ) {

        m_dimension.print( os );
        os << " ";
        printImplementationType( os, m_implementation_type );
        if ( m_implementation_type == METAL ) {        
            os << " ";
            printMetalImplementationType( os, m_metal_implementation_type );
        }
        os << " ";
        m_execution_configuration.print( os );
        return os;
    }

    virtual ~TestCaseConfiguration(){;}
};

template<class T>
static inline void static_type_guard() {
    static_assert(
           is_same< short,T >::value
        || is_same< int,  T >::value 
        || is_same< long, T >::value
        || is_same< float,T >::value
        || is_same< double,T >::value  );
}


template<class T>
static inline void static_type_guard_real() {
    static_assert( is_same< float,T >::value || is_same< double,T >::value );
}


class TestCaseWithTimeMeasurements {

  protected:
    TestCaseConfiguration m_configuration;

    bool   m_verification_true_false;
    double m_verification_rms;
    double m_verification_dist;

    string         m_type_string;
    vector<double> m_measured_times;
    double         m_mean_times;
    double         m_stddev_times;

  public:

    TestCaseWithTimeMeasurements()
        :m_verification_true_false ( false )
        ,m_verification_rms        ( 0.0   )
        ,m_verification_dist       ( 0.0   )
        ,m_mean_times              ( 0.0   )
        ,m_stddev_times            ( 0.0   )
    {;}

    virtual ~TestCaseWithTimeMeasurements(){;}

    void setDataElementType(const TestDataElementType data_type ) {
        m_configuration.setDataElementType( data_type );
    }
    void setVector( const int dimension ) {
        m_configuration.setVector(dimension);
    }
    void setMatrixColMajor( const int row, const int col ) {
        m_configuration.setMatrixColMajor( row, col );
    }
    void setMatrixRowMajor( const int row, const int col ) {
        m_configuration.setMatrixRowMajor( row, col );
    }
    void setMatrixSparse( const int row, const int col, const int nnz ) {
        m_configuration.setMatrixSparse( row, col, nnz );
    }
    void setSOA( const int dimension ) {
        m_configuration.setSOA(dimension);
    }
    void setAOS( const int dimension ) {
        m_configuration.setAOS(dimension);
    }
    void setRandomDiagonallyDominantSkewSymmetric( const int dimension ) {
        m_configuration.setRandomDiagonallyDominantSkewSymmetric( dimension );
    }
    void setRandomDiagonallyDominantSymmetric( const int dimension ) {
        m_configuration.setRandomDiagonallyDominantSymmetric( dimension );
    }
    void setRealNonsymmetricMu02( const int dimension ) {
        m_configuration.setRealNonsymmetricMu02( dimension );
    }
    void setRealNonsymmetricMu08( const int dimension ) {
        m_configuration.setRealNonsymmetricMu08( dimension );
    }
    void setRealSymmetric( const int dimension ) {
        m_configuration.setRealSymmetric( dimension );
    }
    void setVerificationType( const TestDataVerificationType t ) {
        m_configuration.setVerificationType( t );
    }
    void setMemcpy( const int num_threads ) {
        m_configuration.setMemcpy( num_threads );
    }
    void setCPPBlock( const int num_threads, const int factor_loop_unrolling ) {
        m_configuration.setCPPBlock( num_threads, factor_loop_unrolling );
    }
    void setColumnCholesky() {
        m_configuration.setColumnCholesky();
    }
    void setSubmatrixCholesky() {
        m_configuration.setSubmatrixCholesky();
    }
    void setCPPInterleaved( const int num_threads, const int factor_loop_unrolling ) {
        m_configuration.setCPPInterleaved( num_threads, factor_loop_unrolling );
    }
    void setNEON( const int num_threads, const int factor_loop_unrolling ) {
        m_configuration.setNEON( num_threads, factor_loop_unrolling );
    }

    void setBoostSampleSort( const int num_threads ) {
        m_configuration.setBoostSampleSort( num_threads );
    }        

    void setMetal( const MetalImplementationType t, const int groups_per_grid, const int threads_per_group ) {
        m_configuration.setMetal( t, groups_per_grid, threads_per_group );
    }        

    void setImplementationType( const ImplementationType t ) {
        m_configuration.setImplementationType( t );
    }

    void addTime( const double microseconds ) {
        m_measured_times.push_back( microseconds );
    }

    void setTrueFalse( const bool t ) {
        m_verification_true_false = t;
    }

    void setRMS( const double rms) {
        m_verification_rms = rms;
    }

    void setDist( const double dist) {
        m_verification_dist = dist;
    }

    void calculateMeanStddevOfTime() {

        m_mean_times = 0.0;

        const double len = m_measured_times.size();

        for ( auto v : m_measured_times ) {
            m_mean_times += v;
        }

        m_mean_times /= len;

        m_stddev_times = 0.0;

        for ( auto v : m_measured_times ) {

            const double diff = v - m_mean_times;
            const double sq   = diff * diff;
            m_stddev_times += sq;
        }
        m_stddev_times /= ( len - 1 );
    }

    static void printHeader(ostream& os) {
        TestCaseConfiguration::printHeader( os );
        os << "\t";
        os << "vefirication value";
        os << "\t";
        os << "mean time milliseconds";
        os << "\t";
        os << "stddev milliseconds";
        os << "\n";
    }

    virtual ostream& printBrief( ostream& os ) {
        m_configuration.printBrief( os );
        return os;
    }

    virtual void printExtra(ostream& os) { }

    virtual ostream& print( ostream& os ) {
        os << setprecision(8);
        m_configuration.print( os );
        os << "\t";
        switch ( m_configuration.verificationType() ) {
          case TRUE_FALSE:
            os << ( m_verification_true_false ? "TRUE" : "FALSE" );
            break;
          case RMS:
            os << m_verification_rms;
            break;
          case DISTANCE:
            os << m_verification_dist;
            break;
          default:
            break;
        }
        os << "\t";
        os << ( m_mean_times   * 1000.0 );
        os << "\t";
        os << ( m_stddev_times * 1000.0 );
        printExtra( os );
        os << "\n";

        return os;
    }

    virtual void run() = 0;

    virtual void prologue() {}
    virtual void epilogue() {}
};


class TestExecutor {

  protected:
    vector< shared_ptr< TestCaseWithTimeMeasurements > > m_test_cases;

    const int m_num_trials;
    ostream&  m_os;

  public:
    TestExecutor( ostream& os, const int num_trials )
        :m_num_trials( num_trials ) 
        ,m_os(os){;}

    virtual ~TestExecutor(){;}

    void addTestCase( shared_ptr< TestCaseWithTimeMeasurements>&& c ) {
        m_test_cases.emplace_back( c );
    }

    virtual void   prepareForBatchRuns   ( const int test_case ){;}
    virtual void   cleanupAfterBatchRuns ( const int test_case ){;}
    virtual void   prepareForRun         ( const int test_case, const int num ){;}
    virtual void   cleanupAfterRun       ( const int test_case, const int num ){;}


    void execute() {

        for ( int i = 0; i < m_test_cases.size(); i++ ) {

            auto test_case = m_test_cases[i];

            cerr << "Testing [";
            test_case->printBrief( cerr );
            cerr << "] ";

            prepareForBatchRuns(i);

            for ( int j = 0; j < m_num_trials + 1; j++ ) {

                cerr << "." << flush;

                prepareForRun(i, j);

                test_case->prologue();

                auto time_begin = chrono::high_resolution_clock::now();        

                test_case->run();

                auto time_end = chrono::high_resolution_clock::now();        

                test_case->epilogue();

                cleanupAfterRun(i, j);

                chrono::duration<double> time_diff = time_end - time_begin;

                if (j > 0) {
                    // discard the first run.
                    test_case->addTime( time_diff.count() );
                }
            }
            cerr << "\n";

            cleanupAfterBatchRuns(i);
           
        }

        for ( int i = 0; i < m_test_cases.size(); i++ ) {

            auto t = m_test_cases[i];

            t->calculateMeanStddevOfTime();

            t->print( m_os );
        }
        m_os << "\n";
    }
};


template<class T>
inline double getRMSDiffTwoVectors( const T* const a, const T* const b, const size_t num )
{
    static_assert( is_same<float, T>::value || is_same<double, T>::value );

    double diff_sum = 0.0;
    for ( int i = 0; i < num; i++ ) {

        double diff = a[i] - b[i];

	diff_sum += (diff*diff);
    }
    double rms = sqrt(diff_sum / ((double)num));
    return rms;
}


template<class T>
inline double getDistTwoVectors( const T* const a, const T* const b, const size_t num )
{
    static_assert( is_same<float, T>::value || is_same<double, T>::value );

    double diff_sum = 0.0;
    for ( int i = 0; i < num; i++ ) {

        double diff = a[i] - b[i];

	diff_sum += (diff*diff);
    }
    return sqrt(diff_sum);
}


template<class T>
inline bool equalWithinTolerance( const T& v1, const T& v2, const T& tolerance )
{
    const T d = fabs(v1 - v2);

    if ( d <= tolerance ) {
	return true;
    }
    else {
        return ( 2.0*d / (fabs(v1) + fabs(v2)) )  < tolerance;
    }
}


template<class T>
inline T align_up( const T v, const T align ) {

    return ( (v + align - 1) / align ) * align;
}


static inline size_t index_row_major( const size_t x, const size_t y, const size_t width ){
    return y * width + x;
}


template< bool IS_COL_MAJOR >
static inline int linear_index_mat(const int i, const int j, const int M, const int N) {

    if constexpr ( IS_COL_MAJOR ) {
        return j * N + i;
    }
    else {
        return i * M + j;
    }
}



// indexing lower-diagonal matrix in column-major:
//
//     0   1   2   3 <= j 
//   +---+-----------+
// 0 | 0 |           |
//   +---+---+-------+
// 1 | 1 | 4 |       |
//   +---+---+---+---+
// 2 | 2 | 5 | 7 |   |
//   +---+---+---+---+
// 3 | 3 | 6 | 8 | 9 |
//   +---+---+---+---+
// ^
// i
//
//
// indexing lower-diagonal matrix in row-major:
//
//     0   1   2   3 <= j
//   +---+-----------+
// 0 | 0 |           |
//   +---+---+-------+
// 1 | 1 | 2 |       |
//   +---+---+---+---+
// 2 | 3 | 4 | 5 |   |
//   +---+---+---+---+
// 3 | 6 | 7 | 8 | 9 |
//   +---+---+---+---+
// ^
// i

template<bool IS_COL_MAJOR>
static inline int lower_mat_index( const int i, const int j, const int dim ) {

    assert ( i >= j );

    if constexpr (IS_COL_MAJOR) {

        const int num_elems = (dim + 1) * dim / 2;
        const int i_rev = (dim -1) - i;
        const int j_rev = (dim -1) - j;
        return num_elems - 1 - ( j_rev * (j_rev + 1) /2 + i_rev );
    }
    else {
        return (i + 1) * i / 2 + j;
    }
}


#endif /*__TEST_CASE_TIME_MEASUREMENTS_H__*/
