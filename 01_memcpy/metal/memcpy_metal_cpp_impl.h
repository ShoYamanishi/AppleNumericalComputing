#ifndef __MEMCPY_METAL_CPP_IMPL_H__
#define __MEMCPY_METAL_CPP_IMPL_H__

#include <cstddef>

class MemcpyMetalCppImpl
{

  public:
    MemcpyMetalCppImpl( const size_t num_bytes, const bool useManagedBuffer );

    virtual ~MemcpyMetalCppImpl();

    unsigned int numBytes();

    unsigned int numGroupsPerGrid();

    unsigned int numThreadsPerGroup();

    void* getRawPointerIn();

    void* getRawPointerOut();

    void performComputationKernel();

    void performComputationBlit();

  private:
    MemcpyMetalObjC* m_self;

};

#endif /*__MEMCPY_METAL_CPP_IMPL_H__*/
