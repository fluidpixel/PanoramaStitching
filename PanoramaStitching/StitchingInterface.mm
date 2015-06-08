//
//  StitchingInterface.m
//  PanoramaStitching
//
//  Created by Paul on 02/06/2015.
//  Copyright (c) 2015 Fluid Pixel. All rights reserved.
//

#import "StitchingInterface.h"
#import "general_stitching_algorithm.h"

static id<StitchingInterfaceProtocol> staticDelegate;

@implementation StitchingInterface

+(void)setDelegate:(id<StitchingInterfaceProtocol>)delegate;
{
    staticDelegate = delegate;
}

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
    if (staticDelegate) {
        [staticDelegate setText:[NSString stringWithUTF8String:message]];
        [staticDelegate setProgress:NAN];
    }
    //BPLog("CALLBACK: OnInfoReturnCallback: %s, %i\n", message, count);
}
void OnProgressReturnCallback(const char* message, double progress, int count) {
    if (staticDelegate) {
        float progressPercent = 100.0 * progress / float(count);
        [staticDelegate setText:[NSString stringWithFormat:@"%s %0.f%%", message, progressPercent]];
        [staticDelegate setProgress:progressPercent];
    }
}
void OnResetCallback() {
    [staticDelegate setText:@"Reset"];
    //BPLog("CALLBACK: OnResetCallback:\n");
}
void OnImageAddedCallback(const char* path, int i) {
    [staticDelegate setText:[NSString stringWithFormat:@"Image %i added (%s)", i, path]];
//    BPLog("CALLBACK: BubblePodOnImageAddedCallback: %s, %i\n", path, i);
}
void OnProcessedCallback(const char* resultPath, int images_composited, BubblePodErrorE error) {
    if (staticDelegate) {
        if (error != ERROR_OK) {
            [staticDelegate setText:[NSString stringWithFormat:@"%s, %i\tERROR! : %s", resultPath, images_composited, getBubblepodErrorMessage(error).c_str()]];
        }
        else {
            [staticDelegate setText:[NSString stringWithFormat:@"CALLBACK: OnProcessedCallback: %s, %i", resultPath, images_composited]];
            [staticDelegate didFinishProcessingSuccessfully:[NSString stringWithUTF8String:resultPath]];
        }
    }
}
void OnErrorCallback(BubblePodErrorE error) {
    if (staticDelegate) {
        [staticDelegate setText:[NSString stringWithFormat:@"Error: %s", getBubblepodErrorMessage(error).c_str()]];
    }
    //BPLog("CALLBACK: OnErrorCallback: %s", getBubblepodErrorMessage(error).c_str() );
}