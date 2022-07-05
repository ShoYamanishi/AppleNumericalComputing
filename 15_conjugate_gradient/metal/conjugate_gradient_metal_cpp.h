#ifndef __CONJUGATE_GRADIENT_METAL_CPP_H__
#define __CONJUGATE_GRADIENT_METAL_CPP_H__

class ConjugateGradientMetalCppImpl;

class ConjugateGradientMetalCpp
{

  public:

    ConjugateGradientMetalCpp( const int dim, const int num_threads_per_group, const int max_num_iterations, const float epsilon );

    virtual ~ConjugateGradientMetalCpp();

    float* getRawPointerA();

    float* getRawPointerB();

    float* getX();

    void performComputation();

    int getNumIterations();

  private:
    ConjugateGradientMetalCppImpl* m_impl;

};

#endif /*__CONJUGATE_GRADIENT_METAL_CPP_H__*/

