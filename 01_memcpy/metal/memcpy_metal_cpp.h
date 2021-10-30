#ifndef __MEMCPY_METAL_CPP_H__
#define __MEMCPY_METAL_CPP_H__

class MemcpyMetalCppImpl;

class MemcpyMetalCpp
{

  public:
    MemcpyMetalCpp( const size_t num_bytes, const bool useManagedBuffer );

    virtual ~MemcpyMetalCpp();

    unsigned int numBytes();

    unsigned int numGroupsPerGrid();

    unsigned int numThreadsPerGroup();

    void* getRawPointerIn();

    void* getRawPointerOut();

    void performComputationKernel();

    void performComputationBlit();

  private:
    MemcpyMetalCppImpl* m_impl;

};

#endif /*__MEMCPY_METAL_CPP_H__*/
