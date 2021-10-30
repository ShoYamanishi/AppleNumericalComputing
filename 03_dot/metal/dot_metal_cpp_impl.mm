#import "dot_metal_objc.h"
#import "dot_metal_cpp_impl.h"

DotMetalCppImpl::DotMetalCppImpl( const size_t num_elements, const size_t num_threads_per_group, const size_t num_groups_per_grid, const int reduction_type ){

    m_self = [ [ DotMetalObjC alloc ]
              initWithNumElements : num_elements 
                    ReductionType : reduction_type
               NumThreadsPerGroup : num_threads_per_group
                 NumGroupsPerGrid : num_groups_per_grid  
            ];
}

DotMetalCppImpl::~DotMetalCppImpl(){

    [ (id)m_self dealloc ];
    m_self = nil;
}

float* DotMetalCppImpl::getRawPointerX() {
    return [ (id)m_self getRawPointerX ];
}

float* DotMetalCppImpl::getRawPointerY() {
    return [ (id)m_self getRawPointerY ];
}

float DotMetalCppImpl::getDotXY() {
    return [ (id)m_self getDotXY ];
}

void DotMetalCppImpl::performComputation() {
    return [ (id)m_self performComputation ];
}
