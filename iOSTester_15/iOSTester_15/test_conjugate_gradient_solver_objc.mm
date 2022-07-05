
#import <Foundation/Foundation.h>
#import "test_conjugate_gradient_solver_objc.h"

int run_test();

@implementation TestConjugateGradientSolverObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
