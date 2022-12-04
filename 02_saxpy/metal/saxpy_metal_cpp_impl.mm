#import "saxpy_metal_objc.h"
#import "saxpy_metal_cpp_impl.h"

SaxpyMetalCppImpl::SaxpyMetalCppImpl( const size_t num_elements, const size_t num_threads_per_group, const size_t num_groups_per_grid ){

    m_self = [ [ SaxpyMetalObjC alloc ]

              initWithNumElements : num_elements 
               NumThreadsPerGroup : num_threads_per_group
                 NumGroupsPerGrid : num_groups_per_grid  
             ];
}

SaxpyMetalCppImpl::~SaxpyMetalCppImpl(){;}


float* SaxpyMetalCppImpl::getRawPointerX() {

    return [ m_self getRawPointerX ];
}

float* SaxpyMetalCppImpl::getRawPointerY() {

    return [ m_self getRawPointerY ];
}

void SaxpyMetalCppImpl::setScalar_a( const float a ) {

    [ m_self setScalar_a:a ];
}

void SaxpyMetalCppImpl::performComputation() {

    return [ m_self performComputation ];
}
