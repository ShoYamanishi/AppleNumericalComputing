#ifndef __CONJUGATE_GRADIENT_METAL_CPP_IMPL_H__
#define __CONJUGATE_GRADIENT_METAL_CPP_IMPL_H__

#include <cstddef>

class ConjugateGradientMetalCppImpl
{
  public:

    ConjugateGradientMetalCppImpl( const int dim, const int num_threads_per_group, const int max_num_iterations, const float epsilon );

    ~ConjugateGradientMetalCppImpl();

    float* getRawPointerA();

    float* getRawPointerB();

    void   performComputation();

    float* getX();

    int getNumIterations();

  private:
    ConjugateGradientMetalObjC* m_self;
};

#endif /*__CONJUGATE_GRADIENT_METAL_CPP_IMPL_H__*/
