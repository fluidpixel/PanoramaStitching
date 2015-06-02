//
//  StitchingInterface.m
//  PanoramaStitching
//
//  Created by Paul on 02/06/2015.
//  Copyright (c) 2015 Fluid Pixel. All rights reserved.
//

#import "StitchingInterface.h"
#import "general_stitching_algorithm.h"

@implementation StitchingInterface
+(void)initaliseAlgorithm:(int)alg withResultPath:(NSString*)resultPath withPreviewPath:(NSString*)previewPath;
{
    reset();
    initialiseAlgorithm( (int)alg, (char*)[resultPath UTF8String], (char*)[previewPath UTF8String]);
}

+(void)addImagePath:(NSString*)imagePath atIndex:(NSInteger)index isLastImage:(BOOL)isLast;
{
    addImagePath([imagePath UTF8String], (int)index, [[NSNumber numberWithBool:isLast] intValue]);
}

+(void)addImagePath:(NSString*)imagePath atIndex:(NSInteger)index isLastImage:(BOOL)isLast withAttitudeX:(double)x y:(double)y z:(double)z w:(double)w;
{
    addImagePathWithAttitude([imagePath UTF8String], (int)index, [[NSNumber numberWithBool:isLast] intValue], w, x, y, z);
}

@end

std::string getBubblepodErrorMessage(BubblePodErrorE error) {
    switch (error) {
        case ERROR_OK:                  return "Algorithm ran successfully!";
        case ERROR_INCOMPLETE:          return "Algorithm ran successfully but did not use all of the pictures provided";
        case ERROR_NOT_ENOUGH_FEATURES: return "Algorithm could not run due to lack of features";
        case ERROR_ALGORITHM_ERROR:     return "Internal Algorithm Error";
        case ERROR_UNKNOWN:             return "Unknown Error";
        case ERROR_NEED_MORE_IMAGES:    return "Need more images";
        default:                        return "Unknown Error Type";
    }
}

void OnInfoReturnCallback(const char* message, int count) {
    BPLog("CALLBACK: OnInfoReturnCallback: %s, %i\n", message, count);
}
void BubblePodOnInfoReturnCallback(const char* message, int count) {
    BPLog("CALLBACK: BubblePodOnInfoReturnCallback: %s, %i\n", message, count);
}
void OnResetCallback() {
    BPLog("CALLBACK: OnResetCallback:\n");
}
void OnImageAddedCallback(const char* path, int i) {
    BPLog("CALLBACK: BubblePodOnImageAddedCallback: %s, %i\n", path, i);
}
void OnProcessedCallback(const char* resultPath, int images_composited, BubblePodErrorE error) {
    if (error)  {
        BPLog("CALLBACK: OnProcessedCallback: %s, %i\tERROR! : %s", resultPath, images_composited, getBubblepodErrorMessage(error).c_str() );
    }
    else {
        BPLog("CALLBACK: OnProcessedCallback: %s, %i", resultPath, images_composited );
    }
}
void OnErrorCallback(BubblePodErrorE error) {
    BPLog("CALLBACK: OnErrorCallback: %s", getBubblepodErrorMessage(error).c_str() );
}