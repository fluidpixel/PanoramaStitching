//
//  BPRefiner.h
//  BubblePix
//
//  Created by Paul on 09/07/2014.
//  Copyright (c) 2014 Fluid Pixel. All rights reserved.
//

// based on code from https://github.com/Itseez/opencv/blob/master/modules/stitching/include/opencv2/stitching/detail/motion_estimators.hpp
// downloaded on 9 July 2014
// copyright message from original code is below:

/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                          License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/


// Functions and algorithms are arranged into a single class without virtual functions
// algorithms themselves are the same as before.


#ifndef __BubblePix__BPRefiner__
#define __BubblePix__BPRefiner__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2//stitching/detail/util.hpp>
#include <opencv2/stitching/detail/matchers.hpp>


// Return struct form FindMaxSpanningTree function
using cv::detail::ImageFeatures;
using cv::detail::MatchesInfo;
using cv::detail::CameraParams;
using cv::detail::Graph;
using std::vector;
using cv::Mat;

struct SpanningTreeRV {
    Graph span_tree;
    vector<int> centers;
};

class BPRefiner {
public:
    BPRefiner(int num_params_per_cam = 4, int num_errs_per_measurement = 3);
    
    bool operator ()(const vector<ImageFeatures> &features, const vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras);
    

    void setConfThresh(double conf_thresh) {
        conf_thresh_ = conf_thresh;
    }
    void setTermCriteria(const cv::TermCriteria& term_criteria) {
        term_criteria_ = term_criteria;
    }
    
    double confThresh() const {
        return conf_thresh_;
    }
    cv::TermCriteria termCriteria() const {
        return term_criteria_;
    }
    
    void setMaxIterations(unsigned int value) {
        max_iterations = value;
    }
    unsigned int maxIterations() {
        return max_iterations;
    }
    
private:
    void setUpInitialCameraParams(const vector<CameraParams> &cameras);
    void obtainRefinedCameraParams(vector<CameraParams> &cameras) const;
    void calcError(Mat &err);
    void calcJacobian(Mat &jac);
    
    SpanningTreeRV findMaxSpanningTree(int num_images, const vector<MatchesInfo> &pairwise_matches);
    void calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res);
    
    static std::vector< std::pair< int, int> > calculate_ranges(const std::vector<int> & s_tree, int num_items);
    
    Mat err1_, err2_;
    
    int num_images_;
    int total_num_matches_;
    
    int num_params_per_cam_;
    int num_errs_per_measurement_;
    
    const ImageFeatures *features_;
    const MatchesInfo *pairwise_matches_;
    
    // Threshold to filter out poorly matched image pairs
    double conf_thresh_;
    
    //Levenberg–Marquardt algorithm termination criteria
    cv::TermCriteria term_criteria_;
    
    // Camera parameters matrix (CV_64F)
    Mat cam_params_;
    
    // Connected images pairs
    vector<int> edges_;
    
    
    // Limit the number of iterations on the main loop;
    unsigned int max_iterations;
    
};



#endif /* defined(__BubblePix__BPRefiner__) */
