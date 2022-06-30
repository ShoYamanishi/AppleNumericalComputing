//
//  test_lcp_objc.m
//  iOSTester_14
//
//  Created by Shoichiro Yamanishi on 15.06.22.
//

#import <Foundation/Foundation.h>
#import "test_lcp_objc.h"

int run_test( const char * );

@implementation TestLCPObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    
    NSString * filePath = [[NSBundle mainBundle] pathForResource:@"sample_data_32_mu0.2"
                                                          ofType:@"txt"];
    const char* file_path_cstr = [ filePath cStringUsingEncoding:NSUTF8StringEncoding ];

    run_test( file_path_cstr );
}
@end
