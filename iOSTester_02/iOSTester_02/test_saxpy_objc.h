//
//  test_saxpy_objc.h
//  iOSTester_02
//
//  Created by Shoichiro Yamanishi on 19.04.22.
//

#ifndef test_saxpy_objc_h
#define test_saxpy_objc_h

#import <Foundation/Foundation.h>
#import <simd/simd.h>

@interface TestSaxpyObjc: NSObject
-(instancetype) init;
-(void) run;
@end

#endif /* test_saxpy_objc_h */
