//
//  test_saxpy_objc.m
//  iOSTester_02
//
//  Created by Shoichiro Yamanishi on 19.04.22.
//

#import <Foundation/Foundation.h>

#import "test_saxpy_objc.h"

int run_test();

@implementation TestSaxpyObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
