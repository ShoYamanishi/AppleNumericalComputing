//
//  test_cholesky_decomp_objc.m
//  iOSTester_10
//
//  Created by Shoichiro Yamanishi on 20.04.22.
//
#import <Foundation/Foundation.h>
#import "test_cholesky_decomp_objc.h"

int run_test();

@implementation TestCholeskyDecompObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
