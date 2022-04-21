//
//  test_fft_objc.m
//  iOSTester_13
//
//  Created by Shoichiro Yamanishi on 21.04.22.
//

#import <Foundation/Foundation.h>
#import "test_fft_objc.h"

int run_test();

@implementation TestFFTObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
