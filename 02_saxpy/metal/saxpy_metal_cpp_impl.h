#ifndef __SAXPY_METAL_CPP_IMPL_H__
#define __SAXPY_METAL_CPP_IMPL_H__

#include <cstddef>

class SaxpyMetalCppImpl
{

  public:
    SaxpyMetalCppImpl( const size_t num_elements, const size_t num_threads_per_group, const size_t num_groups_per_grid );

    virtual ~SaxpyMetalCppImpl();

    float* getRawPointerX();

    float* getRawPointerY();

    void setScalar_a( const float a );

    void performComputation();

  private:
    SaxpyMetalObjC* m_self;

};

#endif /*__SAXPY_METAL_CPP_IMPL_H__*/
