#import "saxpy_metal_objc.h"
#import "saxpy_metal_cpp_impl.h"

SaxpyMetalCppImpl::SaxpyMetalCppImpl( const size_t num_elements, const size_t num_threads_per_group, const size_t num_groups_per_grid ){

    m_self = [ [ SaxpyMetalObjC alloc ]

              initWithNumElements : num_elements 
               NumThreadsPerGroup : num_threads_per_group
                 NumGroupsPerGrid : num_groups_per_grid  
             ];
}

SaxpyMetalCppImpl::~SaxpyMetalCppImpl(){

    [ (id)m_self dealloc ];
    m_self = nil;
}

float* SaxpyMetalCppImpl::getRawPointerX() {

    return [ (id)m_self getRawPointerX ];
}

float* SaxpyMetalCppImpl::getRawPointerY() {

    return [ (id)m_self getRawPointerY ];
}

void SaxpyMetalCppImpl::setScalar_a( const float a ) {

    [ (id)m_self setScalar_a:a ];
}

void SaxpyMetalCppImpl::performComputation() {

    return [ (id)m_self performComputation ];
}

