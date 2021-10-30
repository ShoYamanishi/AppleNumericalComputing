#ifndef __DOT_METAL_CPP_IMPL_H__
#define __DOT_METAL_CPP_IMPL_H__

#include <cstddef>

class DotMetalCppImpl
{

  public:
    DotMetalCppImpl( const size_t num_elements, const size_t num_threads_per_group, const size_t num_groups_per_grid, const int reduction_type );

    virtual ~DotMetalCppImpl();

    float* getRawPointerX();

    float* getRawPointerY();

    float  getDotXY();

    void performComputation();

  private:
    void * m_self;

};

#endif /*__DOT_METAL_CPP_IMPL_H__*/
