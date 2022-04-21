//
//  test_radix_sort_objc.m
//  iOSTester_05
//
//  Created by Shoichiro Yamanishi on 19.04.22.
//

#import <Foundation/Foundation.h>

#import "test_radix_sort_objc.h"

int run_test();

@implementation TestRadixSortObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
