//
//  BPRefiner.cpp
//  BubblePix
//
//  Created by Paul on 09/07/2014.
//  Copyright (c) 2014 Fluid Pixel. All rights reserved.
//


// based on code from https://github.com/Itseez/opencv/blob/master/modules/stitching/src/motion_estimators.cpp
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


#include "BPRefiner.h"

using cv::detail::ImageFeatures;
using cv::detail::MatchesInfo;
using cv::detail::CameraParams;
using cv::detail::Graph;
using std::vector;
using cv::Mat;


struct IncDistance {
    IncDistance(std::vector<int> &vdists) : dists(&vdists[0]) {}
    void operator ()(const cv::detail::GraphEdge &edge) {
        dists[edge.to] = dists[edge.from] + 1;
    }
    int* dists;
};


BPRefiner::BPRefiner(int num_params_per_cam, int num_errs_per_measurement):   num_params_per_cam_(num_params_per_cam),
num_errs_per_measurement_(num_errs_per_measurement), max_iterations(~0)
{
    setConfThresh(1.);
    setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, DBL_EPSILON));
}

SpanningTreeRV BPRefiner::findMaxSpanningTree( int num_images, const std::vector<MatchesInfo> &pairwise_matches) {
    
    SpanningTreeRV rv;
    
    Graph graph(num_images);
    std::vector<cv::detail::GraphEdge> edges;
    
    // Construct images graph and remember its edges
    for (int k = 0; k < pairwise_matches.size(); k++) {
        MatchesInfo matches = pairwise_matches[k];
        
        if (matches.H.empty())
            continue;
        float conf = (float)(matches.num_inliers);
        const int i = matches.src_img_idx;
        const int j = matches.dst_img_idx;
        graph.addEdge(i, j, conf);
        edges.push_back( cv::detail::GraphEdge(i, j, conf) );
    }
    
    cv::detail::DisjointSets comps(num_images);
    rv.span_tree.create(num_images);
    std::vector<int> span_tree_powers(num_images, 0);
    
    // Find maximum spanning tree
    sort(edges.begin(), edges.end(), std::greater<cv::detail::GraphEdge>());
    for (size_t i = 0; i < edges.size(); ++i)
    {
        int comp1 = comps.findSetByElem(edges[i].from);
        int comp2 = comps.findSetByElem(edges[i].to);
        if (comp1 != comp2)
        {
            comps.mergeSets(comp1, comp2);
            rv.span_tree.addEdge(edges[i].from, edges[i].to, edges[i].weight);
            rv.span_tree.addEdge(edges[i].to, edges[i].from, edges[i].weight);
            span_tree_powers[edges[i].from]++;
            span_tree_powers[edges[i].to]++;
        }
    }
    
    // Find spanning tree leafs
    std::vector<int> span_tree_leafs;
    for (int i = 0; i < num_images; ++i)
        if (span_tree_powers[i] == 1)
            span_tree_leafs.push_back(i);
    
    // Find maximum distance from each spanning tree vertex
    std::vector<int> max_dists(num_images, 0);
    std::vector<int> cur_dists;
    for (size_t i = 0; i < span_tree_leafs.size(); ++i)
    {
        cur_dists.assign(num_images, 0);
        rv.span_tree.walkBreadthFirst( span_tree_leafs[i], IncDistance(cur_dists) );
        for (int j = 0; j < num_images; ++j) max_dists[j] = std::max(max_dists[j], cur_dists[j]);
    }
    
    // Find min-max distance
    int min_max_dist = max_dists[0];
    for (int i = 1; i < num_images; ++i) if (min_max_dist > max_dists[i]) min_max_dist = max_dists[i];
    
    // Find spanning tree centers
    rv.centers.clear();
    for (int i = 0; i < num_images; ++i) if (max_dists[i] == min_max_dist) rv.centers.push_back(i);

    // CV_Assert( rv.centers.size() > 0 && rv.centers.size() <= 2);
    
    return rv;
}


std::vector< std::pair< int, int> > BPRefiner::calculate_ranges(const std::vector<int> & s_tree, int num_items) {
    // takes a list of indices (must be sorted in increasing order) and returns a list of paris of the start and end
    // of the ranges. Will wrap-around the size num_items  e.g.
    // s_tree = 0, 2,3, 7,8,9 13, 19,20,21      num_items = 22
    // returns:
    // (2, 3)   (7, 9)   (13, 13)  (19, 0)
    
    std::vector< std::pair< int, int> > ranges;
    if (s_tree.size() == num_items ) {
        ranges.push_back( std::pair<int, int>(0, num_items - 1) );
        return ranges;
    }
    if (s_tree.size() == 0 ) {
        return ranges;
    }
    if (s_tree.size() == 1 ) {
        ranges.push_back( std::pair<int, int>(s_tree[0], s_tree[0]) );
        return ranges;
    }
    
    int start = s_tree[0];
    int end = s_tree[0];
    for (int i = 1; i < s_tree.size(); i++ ) {
        if ( (s_tree[i-1] + 1) == s_tree[i] ) {
            end = s_tree[i];
        }
        else {
            ranges.push_back( std::pair<int, int>(start, end) );
            start = s_tree[i];
            end = s_tree[i];
        }
    }
    // check for wraparound straddling range
    if (ranges.size() == 0 ) {
        ranges.push_back( std::pair<int, int>(start, end) );
    }
    else if ( (ranges[0].first == 0) && (end == num_items - 1) ) {
        ranges[0].first = start;
    }
    else {
        ranges.push_back( std::pair<int, int>(start, end) );
    }
    
    return ranges;
    
}


bool BPRefiner::operator()(const vector<ImageFeatures> &features,
                            const vector<MatchesInfo> &pairwise_matches,
                            vector<CameraParams> &cameras)
{
    
    num_images_ = static_cast<int>(features.size());
    features_ = &features[0];
    pairwise_matches_ = &pairwise_matches[0];
    
    setUpInitialCameraParams(cameras);
    
    // Leave only consistent image pairs
    edges_.clear();
    total_num_matches_ = 0;
    for (int k = 0; k < pairwise_matches.size(); k++) {
        const MatchesInfo & matches_info = pairwise_matches_[k];
        if (matches_info.confidence > conf_thresh_) {
            edges_.push_back(k);
            total_num_matches_ += (int)( pairwise_matches[k].num_inliers );
        }
        
    }
    
    CvLevMarq solver(num_images_ * num_params_per_cam_, total_num_matches_ * num_errs_per_measurement_, term_criteria_);
    
    Mat err, jac;
    CvMat matParams = cam_params_;
    cvCopy(&matParams, solver.param);
    
    int iter = 0;
    
    //for(;;)
    for(int loop_iteraror = 0;loop_iteraror < max_iterations; loop_iteraror++)
    {
        const CvMat* _param = 0;
        CvMat* _jac = 0;
        CvMat* _err = 0;
        
        bool proceed = solver.update(_param, _jac, _err);
        
        cvCopy(_param, &matParams);
        
        if (!proceed || !_err)
            break;
        
        if (_jac) {
            calcJacobian(jac);
            CvMat tmp = jac;
            cvCopy(&tmp, _jac);
        }
        
        if (_err) {
            calcError(err);
            iter++;
            CvMat tmp = err;
            cvCopy(&tmp, _err);
        }
    }
    
    // Check if all camera parameters are valid
    for (int i = 0; i < cam_params_.rows; ++i) {
        if (cvIsNaN(cam_params_.at<double>(i,0))) {
            std::cout << "\nCamera parameter not valid\n";
            return false;
        }
    }
    
    obtainRefinedCameraParams(cameras);
    
    SpanningTreeRV stree = findMaxSpanningTree(num_images_, pairwise_matches);
    
    if (stree.centers.size() > 0) {
        Mat R_inv = cameras[stree.centers[0]].R.inv();
        for (int i = 0; i < num_images_; ++i) cameras[i].R = R_inv * cameras[i].R;
        if ( stree.centers.size() == num_images_ ) {
            // ignored
        }
        else if ( stree.centers.size() > 2 ) {
            
            std::vector< std::pair< int, int> > ranges = calculate_ranges(stree.centers, num_images_);

            if ( ranges.size() > 0 ) {
                for (int i = 0; i < ranges.size(); i++) {
                    int start = ranges[i].first;
                    int end = ranges[i].second + 1;
                    if (end < start) end += num_images_;
                    
                    double focal = cameras[(start + num_images_ - 1) % num_images_].focal;
                    double focalD = (cameras[end % num_images_].focal - focal) / double (end - start + 1);
                    int j = start;
                    for (; j < end; j++, focal += focalD) cameras[j % num_images_].focal = focal;
                    
                }
            }
            else {
                std::cout << "\nSomething went wrong\n";
                return false;
            }
        }
        else {
            if (!( stree.centers.size() > 0 && stree.centers.size() <= 2)) {
                std::cout << "\n\nSomething went wrong\n";
                std::cout << "\n stree.centers.size() > 0 && stree.centers.size() <= 2 is false!\n";
                std::cout << "stree.centers.size() = " << stree.centers.size() << "\n";
                return false;
            }
        }
    }
    else {
        std::cout << "\n\nSomething went wrong!\nstree.centers.size() = " << stree.centers.size() << "\n";
        return false;
    }

    return true;
}

void BPRefiner::setUpInitialCameraParams(const std::vector<CameraParams> &cameras) {
    cam_params_.create(num_images_ * 4, 1, CV_64F);
    cv::SVD svd;
    for (int i = 0; i < num_images_; ++i)
    {
        cam_params_.at<double>(i * 4, 0) = cameras[i].focal;
        
        svd(cameras[i].R, cv::SVD::FULL_UV);
        Mat R = svd.u * svd.vt;
        if (determinant(R) < 0)
            R *= -1;
        
        Mat rvec;
        Rodrigues(R, rvec);

        if (rvec.type() == CV_64F) {
            cam_params_.at<double>(i * 4 + 1, 0) = rvec.at<double>(0, 0);
            cam_params_.at<double>(i * 4 + 2, 0) = rvec.at<double>(1, 0);
            cam_params_.at<double>(i * 4 + 3, 0) = rvec.at<double>(2, 0);
            
            std::cout << "\n64 bit camera parameters?: " << rvec.at<double>(0, 0) << "\t" <<
                            rvec.at<double>(1, 0) << "\t" << rvec.at<double>(2, 0) << "\n";
        }
        else {
            if (rvec.type() != CV_32F) {
                Mat tmp;
                rvec.convertTo(tmp, CV_32F);
                rvec = tmp;
                std::cout << "\nunknown camera parameter format. Converted to CV_32F ... : " <<
                                rvec.at<float>(0, 0) << "\t" << rvec.at<float>(1, 0) << "\t" << rvec.at<float>(2, 0) << "\n";
            }
            cam_params_.at<double>(i * 4 + 1, 0) = rvec.at<float>(0, 0);
            cam_params_.at<double>(i * 4 + 2, 0) = rvec.at<float>(1, 0);
            cam_params_.at<double>(i * 4 + 3, 0) = rvec.at<float>(2, 0);
            
        }
        
    }
}

void BPRefiner::obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const {
    for (int i = 0; i < num_images_; ++i)
    {
        cameras[i].focal = cam_params_.at<double>(i * 4, 0);
        
        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, cameras[i].R);
        
        Mat tmp;
        cameras[i].R.convertTo(tmp, CV_32F);
        cameras[i].R = tmp;
    }
}


void BPRefiner::calcError(Mat &err) {
    err.create(total_num_matches_ * 3, 1, CV_64F);
    int match_idx = 0;
    for (size_t edge_index = 0; edge_index < edges_.size(); edge_index++) {
        
        const MatchesInfo& matches_info = pairwise_matches_[ edges_[edge_index] ];
        const int i = matches_info.src_img_idx;
        const int j = matches_info.dst_img_idx;
        
        double f1 = cam_params_.at<double>(i * 4, 0);
        double f2 = cam_params_.at<double>(j * 4, 0);
        
        double R1[9];
        Mat R1_(3, 3, CV_64F, R1);
        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, R1_);
        
        double R2[9];
        Mat R2_(3, 3, CV_64F, R2);
        rvec.at<double>(0, 0) = cam_params_.at<double>(j * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(j * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(j * 4 + 3, 0);
        Rodrigues(rvec, R2_);
        
        const ImageFeatures& features1 = features_[i];
        const ImageFeatures& features2 = features_[j];
        
        cv::Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
        K1(0,0) = f1; K1(0,2) = features1.img_size.width * 0.5;
        K1(1,1) = f1; K1(1,2) = features1.img_size.height * 0.5;
        
        cv::Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
        K2(0,0) = f2; K2(0,2) = features2.img_size.width * 0.5;
        K2(1,1) = f2; K2(1,2) = features2.img_size.height * 0.5;
        
        cv::Mat_<double> H1 = R1_ * K1.inv();
        cv::Mat_<double> H2 = R2_ * K2.inv();
        
        for (size_t k = 0; k < matches_info.matches.size(); ++k)
        {
            if (!matches_info.inliers_mask[k])
                continue;
            
            const cv::DMatch& m = matches_info.matches[k];
            
            cv::Point2f p1 = features1.keypoints[m.queryIdx].pt;
            double x1 = H1(0,0)*p1.x + H1(0,1)*p1.y + H1(0,2);
            double y1 = H1(1,0)*p1.x + H1(1,1)*p1.y + H1(1,2);
            double z1 = H1(2,0)*p1.x + H1(2,1)*p1.y + H1(2,2);
            double len = std::sqrt(x1*x1 + y1*y1 + z1*z1);
            x1 /= len; y1 /= len; z1 /= len;
            
            cv::Point2f p2 = features2.keypoints[m.trainIdx].pt;
            double x2 = H2(0,0)*p2.x + H2(0,1)*p2.y + H2(0,2);
            double y2 = H2(1,0)*p2.x + H2(1,1)*p2.y + H2(1,2);
            double z2 = H2(2,0)*p2.x + H2(2,1)*p2.y + H2(2,2);
            len = std::sqrt(x2*x2 + y2*y2 + z2*z2);
            x2 /= len; y2 /= len; z2 /= len;
            
            double mult = std::sqrt(f1 * f2);
            err.at<double>(3 * match_idx, 0) = mult * (x1 - x2);
            err.at<double>(3 * match_idx + 1, 0) = mult * (y1 - y2);
            err.at<double>(3 * match_idx + 2, 0) = mult * (z1 - z2);
            
            match_idx++;
        }
    }
}

void BPRefiner::calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res) {
    for (int i = 0; i < err1.rows; ++i) res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}

void BPRefiner::calcJacobian(Mat &jac) {
    jac.create(total_num_matches_ * 3, num_images_ * 4, CV_64F);
    
    double val;
    const double step = 1e-3;
    
    for (int i = 0; i < num_images_; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            val = cam_params_.at<double>(i * 4 + j, 0);
            cam_params_.at<double>(i * 4 + j, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 4 + j, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 4 + j));
            cam_params_.at<double>(i * 4 + j, 0) = val;
        }
    }
}
