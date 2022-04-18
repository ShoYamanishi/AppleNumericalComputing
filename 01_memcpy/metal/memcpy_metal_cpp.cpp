class MemcpyMetalObjC;
#include "memcpy_metal_cpp_impl.h"
#include "memcpy_metal_cpp.h"

MemcpyMetalCpp::MemcpyMetalCpp( const size_t num_bytes, const bool useManagedBuffer )
    :m_impl( new MemcpyMetalCppImpl( num_bytes, useManagedBuffer ) )
{
    ;
}

MemcpyMetalCpp::~MemcpyMetalCpp()
{
    delete m_impl;
};

unsigned int MemcpyMetalCpp::numBytes(){ return m_impl->numBytes(); }

unsigned int MemcpyMetalCpp::numGroupsPerGrid(){ return m_impl->numGroupsPerGrid(); }

unsigned int MemcpyMetalCpp::numThreadsPerGroup(){ return m_impl->numThreadsPerGroup(); }

void* MemcpyMetalCpp::getRawPointerIn()
{
    return m_impl->getRawPointerIn();
}

void* MemcpyMetalCpp::getRawPointerOut()
{
    return m_impl->getRawPointerOut();
}

void MemcpyMetalCpp::performComputationKernel()
{
    m_impl->performComputationKernel();
}

void MemcpyMetalCpp::performComputationBlit()
{
    m_impl->performComputationBlit();
}
