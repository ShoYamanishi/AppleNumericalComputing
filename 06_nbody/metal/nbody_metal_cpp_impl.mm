#import "nbody_metal_objc.h"
#import "nbody_metal_cpp_impl.h"

NBodyMetalCppImpl::NBodyMetalCppImpl( const size_t num_elements ){

    m_self = [ [ NBodyMetalObjC alloc ] initWithNumElements : num_elements ];
}

NBodyMetalCppImpl::~NBodyMetalCppImpl(){;}

unsigned int NBodyMetalCppImpl::numElements() {
    return [ m_self numElements ];
}

struct particle* NBodyMetalCppImpl::getRawPointerParticles() {
    return [ m_self getRawPointerParticles ];
}

void NBodyMetalCppImpl::performComputationDirectionIsP0ToP1( const bool p0_to_p1 ) {

    return [ m_self performComputationDirectionIsP0ToP1:p0_to_p1 ];
}
