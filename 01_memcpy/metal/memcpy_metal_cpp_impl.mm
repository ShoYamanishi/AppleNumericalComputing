#import "memcpy_metal_objc.h"
#import "memcpy_metal_cpp_impl.h"

MemcpyMetalCppImpl::MemcpyMetalCppImpl( const size_t num_bytes, const bool useManagedBuffer ) {

    m_self = [ [ MemcpyMetalObjC alloc ] initWithNumBytes : num_bytes UseManagedBuffer: useManagedBuffer ];
}

MemcpyMetalCppImpl::~MemcpyMetalCppImpl(){;}

unsigned int MemcpyMetalCppImpl::numBytes() {
    return [ m_self numBytes ];
}

unsigned int MemcpyMetalCppImpl::numGroupsPerGrid() {
    return [ m_self numGroupsPerGrid ];
}

unsigned int MemcpyMetalCppImpl::numThreadsPerGroup() {
    return [ m_self numThreadsPerGroup ];
}

void* MemcpyMetalCppImpl::getRawPointerIn() {
    return [ m_self getRawPointerIn ];
}

void* MemcpyMetalCppImpl::getRawPointerOut() {
    return [ m_self getRawPointerOut ];
}

void MemcpyMetalCppImpl::performComputationKernel() {
    return [ m_self performComputationKernel ];
}

void MemcpyMetalCppImpl::performComputationBlit() {
    return [ m_self performComputationBlit ];
}

