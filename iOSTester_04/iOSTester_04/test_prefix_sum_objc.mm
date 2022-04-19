//
//  test_prefix_sum_objc.m
//  iOSTester_04
//
//  Created by Shoichiro Yamanishi on 19.04.22.
//

#import <Foundation/Foundation.h>

#import "test_prefix_sum_objc.h"

int run_test();

@implementation TestPrefixSumObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
