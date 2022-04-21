//
//  test_sparse_matrix_vector_objc.m
//  iOSTester_08
//
//  Created by Shoichiro Yamanishi on 20.04.22.
//

#import <Foundation/Foundation.h>
#import "test_sparse_matrix_vector_objc.h"

int run_test();

@implementation TestSparseMatrixVectorObjc: NSObject
-(instancetype) init
{
    self = [super init];
    return self;
}

-(void) run{
    run_test();
}
@end
