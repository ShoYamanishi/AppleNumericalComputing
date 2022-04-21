//
//  test_gauss_seldel_solver_objc.m
//  iOSTester_12
//
//  Created by Shoichiro Yamanishi on 21.04.22.
//

#import <Foundation/Foundation.h>
#import "test_gauss_seidel_solver_objc.h"

int run_test();

@implementation TestGaussSeidelSolverObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
