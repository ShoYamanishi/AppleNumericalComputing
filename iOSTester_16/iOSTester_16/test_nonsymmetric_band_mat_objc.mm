#import <Foundation/Foundation.h>
#import "test_nonsymmetric_band_mat_objc.h"

int run_test();

@implementation TestNonsymmetricBandMatObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
