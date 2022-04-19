//
//  test_dot_objc.m
//  iOSTester_03
//
//  Created by Shoichiro Yamanishi on 19.04.22.
//

#import <Foundation/Foundation.h>

#import "test_dot_objc.h"

int run_test();

@implementation TestDotObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
