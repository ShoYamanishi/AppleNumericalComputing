#import "conjugate_gradient_metal_objc.h"
#import "conjugate_gradient_metal_cpp_impl.h"

ConjugateGradientMetalCppImpl::ConjugateGradientMetalCppImpl( const int dim, const int num_threads_per_group, const int max_num_iterations, const float epsilon )
{
    m_self = [ [ ConjugateGradientMetalObjC alloc ]
              initWithNumElements: dim
               NumThreadsPerGroup: num_threads_per_group
                 MaxNumIterations: max_num_iterations
                          Epsilon: epsilon                 ];
}

ConjugateGradientMetalCppImpl::~ConjugateGradientMetalCppImpl(){;}

float* ConjugateGradientMetalCppImpl::getRawPointerA() {
    return [ m_self getRawPointerA ];
}

float* ConjugateGradientMetalCppImpl::getRawPointerB() {
    return [ m_self getRawPointerB ];
}

void ConjugateGradientMetalCppImpl::performComputation() {
    return [ m_self performComputation ];
}

float* ConjugateGradientMetalCppImpl::getX() {
    return [ m_self getX ];
}

int ConjugateGradientMetalCppImpl::getNumIterations() {
    return [ m_self getNumIterations ];
}
