#ifndef __TEST_CASE_LCP_H__
#define __TEST_CASE_LCP_H__

#include "test_case_with_time_measurements.h"
#include "test_lcp_pattern_generator.h"

template<class T, bool IS_COL_MAJOR>
class TestCaseLCP : public TestCaseWithTimeMeasurements {

  protected:
    const int           m_dim;
    const T             m_epsilon;
    T*                  m_M;
    T*                  m_q;
    T*                  m_z;
    T*                  m_w;

    const T             m_condition_num;
    int                 m_num_pivots;
    int                 m_num_iterations;
    int                 m_num_subspace_minimizations;

    double              m_feasibility_error;

  public:

    TestCaseLCP( const int dim, const T condition_num, const T epsilon, const LCPTestPatternType p_type )
        :m_dim          ( dim                      )
        ,m_epsilon      ( epsilon                  )
        ,m_M            ( new T [ dim * dim ]      )
        ,m_q            ( new T [ dim ]            )
        ,m_z            ( new T [ dim ]            )
        ,m_w            ( new T [ dim ]            )
        ,m_condition_num( condition_num )
        ,m_num_pivots   ( 0 )
        ,m_num_iterations ( 0 )
        ,m_num_subspace_minimizations ( 0 )
        ,m_feasibility_error( 0.0 )
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
            setMatrixColMajor( dim, dim );
        }
        else {
            setMatrixRowMajor( dim, dim );
        }

        switch (p_type) {

          case LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC:
            setRandomDiagonallyDominantSkewSymmetric( dim );
            break;

          case LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC:
            setRandomDiagonallyDominantSymmetric( dim );
            break;

          case LCP_REAL_NONSYMMETRIC_MU02:
            setRealNonsymmetricMu02( dim );
            break;

          case LCP_REAL_NONSYMMETRIC_MU08:
            setRealNonsymmetricMu08( dim );
            break;

          case LCP_REAL_SYMMETRIC:
            setRealSymmetric( dim );
            break;

          default:
            assert(true);
        }

        setVerificationType( RMS );
    }

    virtual ~TestCaseLCP() {

        delete[] m_M;
        delete[] m_q;
        delete[] m_z;
        delete[] m_w;
    }

    virtual double errorDistFeasibilityAndComplementarity()
    {
        double error = 0.0;
        for ( int i = 0; i < m_dim; i++ ) {
            error += (-1.0 * min ( (double)m_z[i], 0.0 ));
            error += (-1.0 * min ( (double)m_w[i], 0.0 ));
            error += fabs( m_z[i] * m_w[i] );
        }
        return error;
    }

    virtual void setIterations(
        const int num_pivots,
        const int num_iterations,
        const int num_subspace_minimizations )
    {
        m_num_pivots = num_pivots;
        m_num_iterations = num_iterations;
        m_num_subspace_minimizations = num_subspace_minimizations;
    }

    virtual void checkOutput()
    {
        static_assert( is_same< float,T >::value || is_same< double,T >::value );

        m_feasibility_error = errorDistFeasibilityAndComplementarity();

        double sum_sq = 0.0;
        for (int row = 0; row < m_dim ; row++ ) {

            double Mz_q = (double)(m_q[row]);
            for (int col = 0; col < m_dim ; col++ ) {
                Mz_q += ((double)(m_M[row*m_dim + col] * m_z[col]));
            }                   
            double diff = abs( Mz_q - (double)(m_w[row]) );
            sum_sq += (diff*diff);
        }
        double rms = sqrt(sum_sq / m_dim);

        setRMS( rms );
    }

    virtual void setInitialStates( T* M, T* q ) {
        memcpy( m_M, M, sizeof(T) * m_dim * m_dim );
        memcpy( m_q, q, sizeof(T) * m_dim );
    }

    virtual T* getOutputPointer_z() {
        return m_z;
    }

    virtual T* getOutputPointer_w() {
        return m_w;
    }

    virtual void run() = 0;

    virtual void printExtra(ostream& os) {
        os << "\t"
           << m_feasibility_error
           << "\t"
           << m_condition_num
           << "\t"
           << m_num_pivots 
           << "\t"
           << m_num_iterations
           << "\t"
           << m_num_subspace_minimizations;
    }
};

#endif /*__TEST_CASE_LCP_H__*/
