//
//  StitchingInterface.h
//  PanoramaStitching
//
//  Created by Paul on 02/06/2015.
//  Copyright (c) 2015 Fluid Pixel. All rights reserved.
//

#import <Foundation/Foundation.h>

//#import "PanoramaStitching-Swift.h"

#define BPLog(...) { { printf( __VA_ARGS__ ); printf( "\n" ); } }

@interface StitchingInterface : NSObject

+(void)initaliseAlgorithm:(int)alg withResultPath:(NSString*)resultPath withPreviewPath:(NSString*)previewPath;

+(void)addImagePath:(NSString*)imagePath atIndex:(NSInteger)index isLastImage:(BOOL)isLast;

+(void)addImagePath:(NSString*)imagePath atIndex:(NSInteger)index isLastImage:(BOOL)isLast withAttitudeX:(double)x y:(double)y z:(double)z w:(double)w;

@end


typedef enum BP_ERROR {
    ERROR_OK,
    ERROR_NEED_MORE_IMAGES,
    ERROR_INCOMPLETE,
    ERROR_NOT_ENOUGH_FEATURES,
    ERROR_ALGORITHM_ERROR,
    ERROR_UNKNOWN
} BubblePodErrorE;

void OnInfoReturnCallback(const char* message, int count);
void BubblePodOnInfoReturnCallback(const char* message, int count);
void OnResetCallback();
void OnImageAddedCallback(const char* path, int i);
void OnProcessedCallback(const char* resultPath, int images_composited, BubblePodErrorE error);
void OnErrorCallback(BubblePodErrorE error);

