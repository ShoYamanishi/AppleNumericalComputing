#import "memcpy_metal_objc.h"
#import "memcpy_metal_cpp_impl.h"
                       
MemcpyMetalCppImpl::MemcpyMetalCppImpl( const size_t num_bytes, const bool useManagedBuffer ) {

    m_self = [ [ MemcpyMetalObjC alloc ] initWithNumBytes : num_bytes UseManagedBuffer: useManagedBuffer ];
}

MemcpyMetalCppImpl::~MemcpyMetalCppImpl(){;}

unsigned int MemcpyMetalCppImpl::numBytes() {
    return [ (id)m_self numBytes ];
}

unsigned int MemcpyMetalCppImpl::numGroupsPerGrid() {
    return [ (id)m_self numGroupsPerGrid ];
}

unsigned int MemcpyMetalCppImpl::numThreadsPerGroup() {
    return [ (id)m_self numThreadsPerGroup ];
}

void* MemcpyMetalCppImpl::getRawPointerIn() {
    return [ (id)m_self getRawPointerIn ];
}

void* MemcpyMetalCppImpl::getRawPointerOut() {
    return [ (id)m_self getRawPointerOut ];
}

void MemcpyMetalCppImpl::performComputationKernel() {
    return [ (id)m_self performComputationKernel ];
}

void MemcpyMetalCppImpl::performComputationBlit() {
    return [ (id)m_self performComputationBlit ];
}

