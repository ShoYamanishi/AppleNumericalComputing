//
//  test_jacobi_solver_objc.m
//  iOSTester_11
//
//  Created by Shoichiro Yamanishi on 21.04.22.
//
#import <Foundation/Foundation.h>
#import "test_jacobi_solver_objc.h"

int run_test();

@implementation TestJacobiSolverObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
