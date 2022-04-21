//
//  test_2d_filter_objc.m
//  iOSTester_07
//
//  Created by Shoichiro Yamanishi on 20.04.22.
//

#import <Foundation/Foundation.h>
#import "test_2d_filter_objc.h"

int run_test();

@implementation Test2DFilterObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
