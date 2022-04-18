//
//  test_memcpy_bridge.m
//  iOSTester_01
//
//  Created by Shoichiro Yamanishi on 18.04.22.
//

#import <Foundation/Foundation.h>
#import "test_memcpy_objc.h"

int run_test();

@implementation TestMemcpyObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}

@end
