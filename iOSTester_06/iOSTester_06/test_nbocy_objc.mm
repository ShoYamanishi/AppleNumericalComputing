//
//  test_nbocy_objc.m
//  iOSTester_06
//
//  Created by Shoichiro Yamanishi on 19.04.22.
//
#import <Foundation/Foundation.h>

#import "test_nbody_objc.h"

int run_test();

@implementation TestNBodyObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
