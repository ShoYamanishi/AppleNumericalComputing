#ifndef __SAXPY_METAL_CPP_H__
#define __SAXPY_METAL_CPP_H__

class SaxpyMetalCppImpl;

class SaxpyMetalCpp
{

  public:
    SaxpyMetalCpp( const size_t num_elements , const size_t num_threads_per_group, const size_t num_groups_per_grid );

    virtual ~SaxpyMetalCpp();

    float* getRawPointerX();

    float* getRawPointerY();

    void setScalar_a( const float a );

    void performComputation();

  private:
    SaxpyMetalCppImpl* m_impl;

};

#endif /*__SAXPY_METAL_CPP_H__*/
