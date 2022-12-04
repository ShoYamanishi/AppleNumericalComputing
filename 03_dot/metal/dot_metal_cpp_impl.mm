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

DotMetalCppImpl::~DotMetalCppImpl(){;}

float* DotMetalCppImpl::getRawPointerX() {
    return [ m_self getRawPointerX ];
}

float* DotMetalCppImpl::getRawPointerY() {
    return [ m_self getRawPointerY ];
}

float DotMetalCppImpl::getDotXY() {
    return [ m_self getDotXY ];
}

void DotMetalCppImpl::performComputation() {
    return [ m_self performComputation ];
}
