#ifndef __TEST_CASE_CHOLESKY_GSL_H__
#define __TEST_CASE_CHOLESKY_GSL_H__

#include "test_case_cholesky.h"

template<class T, bool IS_COL_MAJOR>
class TestCaseCholesky_gsl : public TestCaseCholesky<T, IS_COL_MAJOR> {

    void* m_Ap; // gsl_matrix
    void* m_xp; // gsl_vector
    void* m_bp; // gsl_vector

  public:

    TestCaseCholesky_gsl( const int dim );

    virtual ~TestCaseCholesky_gsl();

    virtual void setInitialStates( T* A, T* b );

    virtual void compareTruth( const T* const L_baseline, const T* const x_baseline );

    void run();
};


#endif /*__TEST_CASE_CHOLESKY_GSL_H__*/
