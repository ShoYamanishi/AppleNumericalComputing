#import "convolution_2d_ciimage_objc.h"

#include <type_traits>
#include <CoreImage/CoreImage.h>
#include <CoreImage/CIFilter.h>
#include <CoreImage/CIFilterBuiltins.h>
#include <Metal/Metal.h>

@implementation Convolution2D_CIImageObjc
{

    float*            _mInputRawImage;
    float*            _mOutputRawImage;
    int               _mImageWidth;
    int               _mImageHeight;

    int               _mKernelDim;
    CGFloat           _mFilterWeights[49];// Upto 7x7 convolution.

    CIContext*        _mContextArc;
    CGColorSpaceRef   _mColorSpaceRef;
    NSData*           _mInputNSDataArc;
    CGDataProviderRef _mDataProviderRef;
    CGImageRef        _mInputCGImageRef;
}


- (instancetype) initWithWidth:(size_t) width Height:(size_t) height KernelSize:(size_t) kernel UseGPU:(bool) use_gpu
{
    self = [super init];
    if (self)
    {
        _mImageWidth  = width;
        _mImageHeight = height;
        _mKernelDim   = kernel;

        if ( _mKernelDim != 3 && _mKernelDim != 5 && _mKernelDim != 7 ) {
            NSLog(@"Wrong kernel dimension %d.", _mKernelDim );
        }

        if ( use_gpu ) {

            id<MTLDevice> metal_device = MTLCreateSystemDefaultDevice();

            _mContextArc = [ CIContext contextWithMTLDevice: metal_device ];
            if ( _mContextArc == nullptr ) {
                NSLog(@"Failed to create CIContext for GPU.");
            }
        }
        else {
            _mContextArc = [ CIContext contextWithOptions: @{ kCIContextUseSoftwareRenderer : [NSNumber numberWithBool:true] } ];
            if ( _mContextArc == nullptr ) {
                NSLog(@"Failed to create CIContext for CPU.");
            }
        }

        _mColorSpaceRef = CGColorSpaceCreateWithName( kCGColorSpaceLinearGray );

        size_t row_array_num_bytes = _mImageWidth * _mImageHeight * sizeof(float);
        _mInputRawImage = (float*)malloc( row_array_num_bytes );
        if ( _mInputRawImage == nullptr ) {
            NSLog(@"Failed to allocate raw image array.");
        }

        _mOutputRawImage = (float*)malloc( row_array_num_bytes );
        if ( _mOutputRawImage == nullptr ) {
            NSLog(@"Failed to allocate raw image array.");
        }

        _mInputNSDataArc = [ NSData dataWithBytesNoCopy : _mInputRawImage length : row_array_num_bytes ];
        if ( _mInputNSDataArc == nil ) {
            NSLog(@"Failed to create NSData.");
        }

        _mDataProviderRef = CGDataProviderCreateWithCFData( (CFDataRef)_mInputNSDataArc );
        if ( _mDataProviderRef == nil ) {
            NSLog(@"Failed to create CGDataProvider.");
        }

        CGBitmapInfo bitmap_info = kCGImageAlphaNone | kCGBitmapByteOrder32Host | kCGBitmapFloatComponents;

        _mInputCGImageRef
            = CGImageCreate( 
                  _mImageWidth,                 // size_t width
                  _mImageHeight,                // size_t height
                  32,                           // size_t bitsPerComponent (float=32)
                  32,                           // size_t bitsPerPixel == bitsPerComponent for float
                    row_array_num_bytes
                  / _mImageHeight,              // size_t bytesPerRow -> width in pixels * sizeof(float)
                  _mColorSpaceRef,              // CGColorSpaceRef
                  bitmap_info,                  // CGBitmapInfo
                  _mDataProviderRef,            // CGDataProviderRef
                  NULL,                         // no color mapping
                  NO,                           // no interpoloation
                  kCGRenderingIntentDefault     // CGColorRenderingIntent
              );
    }    
    return self;
}


- (void) release_explicit
{
//    CGColorSpaceRelease   ( _mColorSpaceRef   );
//    CGDataProviderRelease ( _mDataProviderRef );
//    CGImageRelease        ( _mInputCGImageRef );

    _mContextArc = nil;
    _mInputNSDataArc = nil;

    if (_mInputRawImage != nullptr ) {
        free( _mInputRawImage );
    }

    if (_mOutputRawImage != nullptr ) {
        free( _mOutputRawImage );
    }
}

- (void) copyToInputBuffer:(const float*)p
{
    memcpy( _mInputRawImage, p, sizeof(float)*_mImageWidth*_mImageHeight );
}

- (void) copyToKernelBuffer:(const float*)p
{
    for ( size_t i = 0; i< _mKernelDim * _mKernelDim; i++ ) {

        _mFilterWeights[i] = p[i];
    }
}

- (float*) getOutputImagePtr
{
    return _mOutputRawImage;
}

- (void) performConvolution
{
    CIImage* input_ciimage_arc = [ CIImage imageWithCGImage: _mInputCGImageRef
                                                    options: @{ kCIImageColorSpace: [NSNull null] } ];
    if ( input_ciimage_arc == nullptr ) {
        NSLog(@"Failed to create CIImage.");
    }
    CIFilter<CIConvolution>* filter_arc;
    if ( _mKernelDim == 3 ) {
        filter_arc = [ CIFilter convolution3X3Filter ];
    }
    else if  ( _mKernelDim == 5 ) {
        filter_arc = [ CIFilter convolution5X5Filter ];
    }
    else {
        filter_arc = [ CIFilter convolution7X7Filter ];
    }
    if ( filter_arc == nullptr ) {
        NSLog(@"Failed to create conv filter.");
    }
    CIVector* filter_vector_arc = [ [CIVector alloc] initWithValues: _mFilterWeights 
                                                              count: _mKernelDim*_mKernelDim ];
    if ( filter_vector_arc == nullptr ) {
        NSLog(@"Failed to create filter_vector.");
    }
    filter_arc.inputImage = input_ciimage_arc;
    filter_arc.bias       = 0.0;
    filter_arc.weights    = filter_vector_arc;
    [ _mContextArc render:filter_arc.outputImage toBitmap: _mOutputRawImage
                                                 rowBytes: sizeof(float)*_mImageWidth
                                                   bounds: CGRectMake( 0, 0, _mImageWidth, _mImageHeight )
                                                   format: kCIFormatLf
                                               colorSpace: _mColorSpaceRef ];
    filter_vector_arc = nil;
    filter_arc        = nil;
    input_ciimage_arc = nil;
}

@end
