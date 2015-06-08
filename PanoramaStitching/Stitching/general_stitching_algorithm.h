#ifndef STITCHING_H_
#define STITCHING_H_

#pragma once

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
//#include "general_callbacks.h"
//#include "general_run_loop.h"
#include "BPRefiner.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

#define MIN_FEATURES_THRESHHOLD 200

// Default command line args

bool try_gpu = false;
double work_megapix = 0.1;
double seam_megapix = 0.01;
double compose_megapix = -1;
float conf_thresh = 0.8f;
bool is_past_adjuster = false;
string features_type = "orb";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to = "/mnt/sdcard/DCIM/BubblePod/Results/Graph.DOT";
string warp_type = "spherical";
int expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.27f;
string seam_find_type = "gc_color";
int blend_type = cv::detail::Blender::MULTI_BAND;
float blend_strength = 1;
string result_name = "result.jpg";
const int max_images = 48;
bool isCapturingHD = false;
bool isRecording = false;
int matchAdjacent = 1;

bool wide_angle = false;

std::stringstream rawAttitude;
std::string save_raw_attitude_to = "raw_attitude.dat";

vector<string> img_names;
vector<ImageFeatures> features_global(max_images);
vector<Mat> images_global(max_images);
vector<cv::Size> full_img_sizes_global(max_images);
vector<CameraParams> cameras_global(max_images);

Ptr<FeaturesFinder> finder;

double work_scale = 1, seam_scale = 1, compose_scale = 1;
bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
double seam_work_aspect = 1;

bool isProcessing = false;

#include <string>
#include <list>
typedef void (*task_fptr)(std::list<int> int_params, std::list<std::string> string_params);
//void addTaskToThread1(task_fptr fptr, std::string taskName, std::list<int> int_params, std::list<std::string> string_params);
void addTaskToThread1(task_fptr fptr, std::string taskName, std::list<int> int_params, std::list<std::string> string_params) {
    fptr(int_params, string_params);
}



static Mat loadAndUndistortWideAngleImage(const std::string & path, int flags = 1) {
    
    if (wide_angle) {
        
        Mat source = imread(path, flags);
        if (source.empty()) {
            return source;
        }
        
        cv::Size sz = source.size();
        
        // use largest axis for scaling
        double scale = (sz.width > sz.height) ? (sz.width) : (scale = sz.height);
        
        // camera intrinsics for scaling of image
        Mat K = (Mat_<double>(3, 3) << scale, 0.0, sz.width * 0.5,  0.0, scale, sz.height * 0.5,  0.0, 0.0, 1.0);
        
        // parameters for undistortion (pincushion/barrel distortion)
        Mat dCo = (Mat_<double>(1, 4) << -0.5, 0.0, 0.0, 0.0);
        
        Mat mapx, mapy;
        
        Mat dest;
        
        // use initUndistortRectifyMap and remap to give more flexibility - will be faster if masks are applied at
        // a later date since initUndistortRectifyMap only needs to run once.
        
        initUndistortRectifyMap(K, dCo, Mat(), K, sz, CV_32FC1, mapx, mapy);
        remap(source, dest, mapx, mapy, INTER_LINEAR, BORDER_TRANSPARENT);
        
        return dest;
    }
    else {
        return imread(path, flags);
    }
}

static int initialise() {
    
    BPLog("Initialising BubblePod Stitching");
    
    // set up save to graph log
    
    isRecording = true;
    if (features_type == "surf") {
#if defined(HAVE_OPENCV_NONFREE) && defined(HAVE_OPENCV_GPU) && !defined(ANDROID)
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            finder = new SurfFeaturesFinderGpu();
        else
#endif
            finder = new SurfFeaturesFinder();
    } else if (features_type == "orb") {
        finder = new OrbFeaturesFinder();
    } else {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        
        OnErrorCallback(ERROR_ALGORITHM_ERROR);
        return -1;
    }
    
#if defined(ANDROID)
    printf("Jprintf loaded finder");
#endif
    
    return 1;
}

static void setIsCapturingHD(int value) {
    isCapturingHD = value;
}

int reset() {
    
    work_scale = 1;
    seam_scale = 1;
    compose_scale = 1;
    is_work_scale_set = false;
    is_seam_scale_set = false;
    is_compose_scale_set = false;
    seam_work_aspect = 1;

    img_names.clear();
    initialise();
    is_past_adjuster = false;
    isProcessing = false;
    
    features_global.clear();
    features_global.resize(max_images);
    
    images_global.clear();
    images_global.resize(max_images);
    
    full_img_sizes_global.clear();
    full_img_sizes_global.resize(max_images);
    
    cameras_global.clear();
    cameras_global.resize(max_images);
    
    rawAttitude.str("");
    
    OnResetCallback();
    BPLog("Reset");
    return 0;
}


int process(char* resultPath) {
    
    
    vector<string> image_paths = img_names;
    img_names.clear();
    int num_images = static_cast<int>(image_paths.size());
    
    features_global.resize(num_images);
    vector<ImageFeatures> features = features_global;
    features_global.clear();
    
    images_global.resize(num_images);
    vector<Mat> images = images_global;
    images_global.clear();
    
    full_img_sizes_global.resize(num_images);
    vector<cv::Size> full_img_sizes = full_img_sizes_global;
    full_img_sizes_global.clear();
    
    cameras_global.resize(num_images);
    vector<CameraParams> cameras = cameras_global;
    cameras_global.clear();
    
    isProcessing = true;
    BPLog("Processing");
    
    int64 app_start_time = getTickCount();
    
    // Check if have enough images
    
    if (num_images < 2) {
        isProcessing = false;
        OnErrorCallback(ERROR_NEED_MORE_IMAGES);
        return -1;
    }
    
    printf("Number of Images: %i\n", num_images);
    
    if ( rawAttitude.str().length() != 0 ) {
        // if present, save camera attitude data
        ofstream f( save_raw_attitude_to.c_str() );
        f << rawAttitude.str();
    }
    
    for (int i =0; i < num_images; i++) {
        if ( cameras[i].R.empty() ) {
            
            // if one is missing, recalculate all camera positions
            std::cout << "Camera roation gyro estimates missing, using image order\n";
            double divAngle = 2.0 * M_PI / (double)num_images;
            for (int j = 0; j < num_images; j++) {
                double yaw	= (double(-j) + 0.5) * divAngle;
                float cosT	= (float)cos(yaw);
                float sinT	= (float)sin(yaw);
                //cameras[j].R = (Mat_<float>(3, 3) << cosT, 0.0, sinT, 0.0, 1.0, 0.0, -sinT, 0.0, cosT);
                //cameras[j].R = (Mat_<float>(3, 3) << cosT, 0.0, -sinT,  sinT, 0.0, cosT, 0.0, -1.0, 0.0);
                cameras[j].R = (Mat_<float>(3, 3) << cosT, 0.0, -sinT,  sinT, 0.0, cosT, 0.0, 1.0, 0.0);
            }
            
            // Use lower confidence threshold for adjuster if exact camera attitudes are not known.
            conf_thresh = 0.0;
            
        }
        
        cameras[i].t= Mat::zeros(1,3, CV_32F);
        // rest of camera parameters estimated later
    }
    
    printf("\nImage Features:\n");
    
    for (int i = 0; i < num_images; i++)
    {
        BPLog("\n %d     %lu   ", i, features[i].keypoints.size());
     
        if (features[i].keypoints.size() < MIN_FEATURES_THRESHHOLD ) {
            if ( features[i].keypoints.size() > 0 ) {
                BPLog("Features deleted (too few)");
            }
            features[i].keypoints.clear();
            features[i].descriptors.release();
        }
    }
    
    BPLog("\nPairwise matching");
    
    OnProgressReturnCallback("Stage 1/3: Finding matches", 0.0, num_images + 2);
    
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    
    // need to check with an even number of images
    if ( (matchAdjacent <= 0) || (matchAdjacent * 2 >= num_images - 1) ) {
        // match every image with every other image
        matcher(features, pairwise_matches);
    }
    else {
        // match only adjacent images separated by at most (matchAdjacent-1) images
        Mat matcher_mask( num_images, num_images, CV_8U, Scalar(0) );
        for (int i = 0; i < num_images; i++) {
            for (int j = matchAdjacent; j >0 ; j--) {
                matcher_mask.at<char>( i, (i+j)%num_images ) = 1;
                matcher_mask.at<char>( (i+j)%num_images, i ) = 1;
            }
        }
        
        matcher(features, pairwise_matches, matcher_mask);
    }
    matcher.collectGarbage();
	   
    // Check if we should save matches graph
    
#ifdef DEBUG
    save_graph = true;
#endif
    
    if (save_graph) {
        printf("\nSaving matches graph...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(image_paths, pairwise_matches, conf_thresh);    // Should this be match_conf?
    }
    
//    vector<double> focals;
//    estimateFocal(features, pairwise_matches, focals);
    
    for (int i = 0; i < num_images; i++) {
        cameras[i].ppx	= 0.5 * features[i].img_size.width;
        cameras[i].ppy	= 0.5 * features[i].img_size.height;
        // use 60° fov as base point
        cameras[i].focal	= features[i].img_size.width * M_SQRT2;
        //cameras[i].focal	= focals[i];
    }
    
    Ptr<BPRefiner> adjuster = new BPRefiner();
    adjuster->setConfThresh(conf_thresh);
    adjuster->setMaxIterations(500);
    
    std::cout << "\nBegin camera refinement (conf_thresh = " << conf_thresh << ")";
    int64 ba_start_time = getTickCount();
    
    double oldFocal = cameras[0].focal;
    
    if (!(*adjuster)(features, pairwise_matches, cameras) ) {
        isProcessing = false;
        OnErrorCallback(ERROR_ALGORITHM_ERROR);
        return -1;
        
    }
    
    bool printFocals = false;
    for (int i = 0; i < num_images; i++) printFocals |= (cameras[i].focal == oldFocal);
    if (printFocals) {
        BPLog("Initial Focal: %f\n", oldFocal);
        for (int i = 0; i < num_images; i++) {
            if (cameras[i].focal == oldFocal) {
                BPLog("Image #%i\t \tfocal:%f \t!\tfeatures: %lu\n", i, cameras[i].focal, features[i].keypoints.size());
            }
            else {
                BPLog("Image #%i\t \tfocal:%f \t \tfeatures: %lu\n", i, cameras[i].focal, features[i].keypoints.size());
            }
        }
    }
    
    std::cout << "Camera refinement took: " << ( getTickCount() - ba_start_time ) / getTickFrequency() << " seconds\n";
    
    features.clear();
    pairwise_matches.clear();
    adjuster.release();
    is_past_adjuster = true;
    
    
    // Find median focal length
    double warped_image_scale;
    {

        vector<double> focals;
        
        for (size_t i = 0; i < cameras.size(); ++i) focals.push_back(cameras[i].focal);
        sort(focals.begin(), focals.end());
        
        if (focals.size() % 2 == 1) warped_image_scale = focals[focals.size() / 2];
        else                        warped_image_scale = (focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
        
    }
    BPLog("\nInitial focal estimate: %f\nMedian focal estimate: %f\n", oldFocal, warped_image_scale);
    
    if (do_wave_correct) {
        
        vector < Mat > rmats;
        
        for (size_t i = 0; i < cameras.size(); ++i)
            
            rmats.push_back(cameras[i].R);
        
        waveCorrect(rmats, wave_correct);
        
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
        
    }
    
    // End of image 'Registration'
    
    
    OnProgressReturnCallback("Stage 2/3: Modifying images", 1.0, num_images + 2);

    
    // Warp images and their masks
    Ptr < WarperCreator > warper_creator;
    
#if defined(HAVE_OPENCV_GPU) && !defined(ANDROID)
    
    if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane") warper_creator = new cv::PlaneWarperGpu();
        else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarperGpu();
        else if (warp_type == "spherical") warper_creator = new cv::SphericalWarperGpu();
    }
    
    else
        
#endif
        
    {
        
        if (warp_type == "plane")                               warper_creator = new cv::PlaneWarper();
        else if (warp_type == "cylindrical")                    warper_creator = new cv::CylindricalWarper();
        else if (warp_type == "spherical")                      warper_creator = new cv::SphericalWarper();
        else if (warp_type == "fisheye")                        warper_creator = new cv::FisheyeWarper();
    }
    
    if (warper_creator.empty()) {
        isProcessing = false;
        
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }
    
    // Estimate memory footprint and rescale if required
    const double APPROX_MAX_MEMORY_FOOTPRINT_MB = 750; // need to check available memory
    const double MEM_EST_MULT = 0.000025;
    const double MAX_NUM_PIXELS = APPROX_MAX_MEMORY_FOOTPRINT_MB / MEM_EST_MULT;
    
    Ptr<RotationWarper> full_size_warper = warper_creator->create( warped_image_scale/work_scale );
    cv::Rect full_size_pano;
    for (int i = 0; i < num_images; ++i) {
        CameraParams tmp = cameras[i];
        tmp.focal /= work_scale;
        tmp.ppx = double(full_img_sizes[i].width) * 0.5;
        tmp.ppy = double(full_img_sizes[i].height) * 0.5;
        
        Mat K;
        tmp.K().convertTo(K, CV_32F);
        cv::Rect warped_roi = full_size_warper->warpRoi(full_img_sizes[i], K, tmp.R);
        
        if (i==0)   full_size_pano = warped_roi;
        else        full_size_pano |= warped_roi;
    }
    
    std::cout << "Memory Footprint Estimated to be: " << double( full_size_pano.size().area() ) * MEM_EST_MULT << "Mb\n";
    double scale_limit = min(1.0, sqrt( MAX_NUM_PIXELS / double( full_size_pano.size().area() ) ) );
    
    if (scale_limit < 1.0) {
        std::cout << "\nOutput panorama too big - images will be rescaled" << scale_limit << "\n";
    }
    if (scale_limit < 0.1) {
        // Don't scale anything below 10%
        scale_limit = 0.1;
    }
    
    
    double new_seam_scale = min(seam_scale, min(512.0 / full_size_pano.height, 2048.0 / full_size_pano.width) );
    if (new_seam_scale < seam_scale) {
        seam_scale = new_seam_scale;
        std::cout << "\nSeam-scale warped images and masks will be too big - must be rescaled\n";
        // clear images loaded during feature finding as we need to rescale them.
        images.clear();
        images.resize(num_images);
    }
    seam_work_aspect = seam_scale / work_scale;
    float float_seam_work_aspect = (float)seam_work_aspect;

    
    vector < Mat > masks_warped(num_images);
    vector < Mat > images_warped(num_images);
    vector < cv::Point > corners(num_images);
    
    Ptr < RotationWarper > warper = warper_creator->create(  float(warped_image_scale) * float_seam_work_aspect);
    
    for (int i = 0; i < num_images; ++i) {
        
        // if the image we loaded during feature finding is still there, use it. Otherwise reload and rescale etc
        Mat img = images[i];
        images[i].release();
        if (img.empty()) {
            img = loadAndUndistortWideAngleImage( image_paths[i] );
            if (!img.empty()) {
                resize(img, img, cv::Size(), seam_scale, seam_scale);
                img.convertTo(img, CV_32F);
            }
        }
        
        // if it is still empty, something has gone wrong
        if (img.empty()) {
            isProcessing = false;
            OnErrorCallback(ERROR_UNKNOWN);
            return -1;
        }

        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        K(0, 0) *= float_seam_work_aspect;
        K(0, 2) *= float_seam_work_aspect;
        K(1, 1) *= float_seam_work_aspect;
        K(1, 2) *= float_seam_work_aspect;
        
        Mat xmap, ymap;
        cv::Rect roi = warper->buildMaps(img.size(), K, cameras[i].R, xmap, ymap);
        remap(img, images_warped[i], xmap, ymap, INTER_LINEAR, BORDER_REFLECT);
        remap(Mat_<uchar>(img.size(), 255), masks_warped[i], xmap, ymap, INTER_NEAREST, BORDER_CONSTANT);
        
        corners[i] = roi.tl();

// FIXME: #216 CV_Assert( cn <= 4 && ssize.area() > 0 ) - likely images[i] is empty/Mat()/images[i].empty() this
//  causes an exception with INTER_LINEAR but not INTER_NEAREST. The latter outputs a 2x2 image; however the
//  corner value returned is (x = -2147483648, y = -2147483648)
//  seam scale is now set to maximum seam-scale panorama size (2048x512) which may, along with the checks
//  above, resolve the problem.
        
    }
    
    images.clear();
    
    //Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    //compensator->feed(corners, images_warped, masks_warped);
    
    Ptr < SeamFinder > seam_finder;
    
    if (seam_find_type == "no")                 seam_finder = new detail::NoSeamFinder();
    else if (seam_find_type == "voronoi")       seam_finder = new detail::VoronoiSeamFinder();
    else if (seam_find_type == "gc_color") {
#if defined(HAVE_OPENCV_GPU) && !defined(ANDROID)
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = new detail::GraphCutSeamFinder( GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad") {
#if defined(HAVE_OPENCV_GPU) && !defined(ANDROID)
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        
        else
#endif
            seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        
    }
    else if (seam_find_type == "dp_color")      seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")  seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
    if (seam_finder.empty()) {
        isProcessing = false;
        cout << "Can't create the following seam finder '" << seam_find_type<< "'\n";
        return 1;
    }
    
    int64 seam_start_time = getTickCount();
    
    seam_finder->find(images_warped, corners, masks_warped);
    seam_finder.release();
    images_warped.clear();
    
    
    std::cout << "\nSeam finding took: " << ( getTickCount() - seam_start_time ) / getTickFrequency() << " seconds\n";

    // Dilate Image masks so they overlap slightly
    for (int i = 0; i < num_images; ++i) {
        Mat dilated_mask;
        dilate(masks_warped[i], dilated_mask, Mat());
        masks_warped[i] = dilated_mask;
    }
    
    OnProgressReturnCallback("Stage 3/3: Stitching", 2.0, num_images + 2);

    
    int64 compose_start_time = getTickCount();
    
    if (!is_compose_scale_set) {
        if (compose_megapix > 0)    compose_scale = min(1.0,sqrt(compose_megapix * 1e6 / full_img_sizes[0].area()));
        else                        compose_scale = 1.0;
        is_compose_scale_set = true;
    }
    if (compose_scale > scale_limit) compose_scale = scale_limit;
    
    // Compute relative scales
    double compose_work_aspect = compose_scale / work_scale;
    
    // Update warped image scale
    warped_image_scale *= compose_work_aspect;
    warper = warper_creator->create( static_cast<float>(warped_image_scale) );
    
    // Update corners and sizes
    cv::Rect destROI;
    for (int i = 0; i < num_images; ++i) {
        // Update intrinsics
        cameras[i].focal *= compose_work_aspect;
        
        // Update corner and size
        cv::Size sz = full_img_sizes[i];
        
        if (std::abs(compose_scale - 1) > 1e-1) {
            sz.width = cvRound(full_img_sizes[i].width * compose_scale);
            sz.height = cvRound(full_img_sizes[i].height * compose_scale);
        }
        
        cameras[i].ppx = sz.width * 0.5;
        cameras[i].ppy = sz.height * 0.5;
        
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        destROI |= warper->warpRoi(sz, K, cameras[i].R);
    }

    float blend_width = sqrt( static_cast<float>( destROI.size().area() ) ) * blend_strength * 0.01f;
    Ptr < Blender > blender;
    if (blend_width < 1.f)
        blender = Blender::createDefault(Blender::NONE, try_gpu);
    else if (blend_type == Blender::MULTI_BAND) {
        int nBands = static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.);
        blender = new MultiBandBlender(try_gpu, nBands);
        cout <<"Multi-band blender, number of bands: "  << nBands;
    }
    else if (blend_type == Blender::FEATHER) {
        float sharpness = 1.f / blend_width;
        blender = new FeatherBlender(sharpness);
        cout << "Feather blender, sharpness: " << sharpness;
    }
    blender->prepare(destROI);
    
    Mat full_img, img;
    Mat xmap, ymap;
    Mat imgWarped, maskWarped, seamMask;
    
    Mat mask;
    
    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        
        // Read image and resize it if necessary
        full_img = loadAndUndistortWideAngleImage( image_paths[img_idx] );
        if (full_img.empty()) {
            isProcessing = false;
            OnErrorCallback(ERROR_UNKNOWN);
            return -1;
        }
        
        if (compose_scale < 0.9)    resize(full_img, img, cv::Size(), compose_scale, compose_scale);
        else                        img = full_img;
        
        full_img.release();
        
        img.convertTo(img, CV_16S);
        
        cv::Size img_size = img.size();
        
        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);
        
        // Build maps once for both image and mask
        
        OnProgressReturnCallback("Stage 3/3: Stitching", float(img_idx + 2.15), num_images + 2);
        cv::Rect roi = warper->buildMaps(img_size, K, cameras[img_idx].R, xmap, ymap);
        OnProgressReturnCallback("Stage 3/3: Stitching", float(img_idx + 2.75), num_images + 2);
        
        // Apply maps to the image
        remap(img, imgWarped, xmap, ymap, INTER_LINEAR, BORDER_REFLECT);
        img.release();
        
        // Apply maps to the mask
        if (mask.empty()) mask = Mat(img_size, CV_8U, Scalar::all(255));
        
        remap( mask, maskWarped, xmap, ymap, INTER_NEAREST, BORDER_CONSTANT);
        xmap.release();
        ymap.release();
                   
        // Process dilated mask
        resize(masks_warped[img_idx], seamMask, maskWarped.size());
        masks_warped[img_idx].release();
        maskWarped &= seamMask;
        seamMask.release();
        
        // Blend the current image
        blender->feed( imgWarped, maskWarped, roi.tl() );
        imgWarped.release();
        maskWarped.release();
        
//        message = "Stage 3/3: Stitching " + std::to_string(img_idx + 1) + " of " + std::to_string(num_images);
//        BubblePodOnInfoReturnCallback((char*)message.c_str(), 0);
        
    }
    
    OnProgressReturnCallback("Stage 3/3: Stitching", float(num_images + 2), num_images + 2);
    
    mask.release();
    
    Mat result, result_mask;
    blender->blend(result, result_mask);
    
    OnInfoReturnCallback((char*) "Stitched", num_images);
    
    cout <<"\nCompositing, time: " << ((getTickCount() - compose_start_time) / getTickFrequency()) << " sec";
    
    isProcessing = false;
    imwrite(resultPath, result);
    
    cout << "\nFinished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency())   << " sec";
    
    OnProcessedCallback(resultPath, num_images, ERROR_OK);
    
    isProcessing = false;
    return 0;
}


static void initialiseAlgorithm(int alg, char* resultPath, char* resultPathPreview) {
    
    BPLog("Initalising Algorithm: %d result path: %s preview path: %s", alg, resultPath, resultPathPreview);
    
    result_name = resultPath;
    
    compose_megapix = -1.f;
    features_type = "surf";
    ba_cost_func = "ray";
    ba_refine_mask = "xxxxx";
    do_wave_correct = true;
    wave_correct = detail::WAVE_CORRECT_HORIZ;
    save_graph = false;
    match_conf = 0.5f; //default 0.5f
    blend_type = Blender::FEATHER;
    
    //intel's options override
    work_megapix = 0.16;
    seam_megapix = 0.08;
    conf_thresh = 0.0;
    warp_type = "cylindrical";
    expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
    seam_find_type = "dp_colorgrad";
    
    if (isCapturingHD)
    {
        compose_megapix = 0.5f;
    }
    
    if (alg == 2) //wide angle
    {
        //conf_thresh = 0.8;
        wide_angle = true;
    }
    else {
        
        wide_angle = false;
        
    }
    
    printf("Initialised Algorithm %d", alg);
    
    initialise();
}

static void process_task(list<int> int_params, list<string> string_params) {
    
    process((char*) string_params.front().c_str());
    
    isProcessing = false;
    
}

static void on_features_found_callback(list<int> int_params,
                                       list<string> string_params) {
    int i = int_params.front();
    string path = string_params.front();
    OnImageAddedCallback((char*) path.c_str(), i);
    
    BPLog("Features found %s int params: %i", (char*)path.c_str(), i);
    
    string_params.clear();
    string_params.push_back(result_name);
    //printf("LASTIMAGEINDEX: %d", int_params.back());
    
    if (int_params.back() == 1) {
        
        string_params.clear();
        string_params.push_back(result_name);
        string_params.push_back("full");
        addTaskToThread1(process_task, "process task", int_params, string_params);

    }
}

static void feature_finding_task(list<int> int_params,
                                 list<string> string_params) {
    
    if (!isRecording)
    {
        BPLog("\nimage rejected (isRecording): %s\n", string_params.front().c_str());
        return;
        
    }
    
    if (int_params.back() == 1)
        isRecording = false;
    
    int i = int_params.front();
    string path = string_params.front();
    
    
    Mat img;
    
    Mat full_img = loadAndUndistortWideAngleImage(img_names[i], 1);
    full_img_sizes_global[i] = full_img.size();
    if (full_img.empty()) {
        BPLog("\nCan't open image %s", path.c_str());
        return;
    }
    if (work_megapix < 0) {
        img = full_img;
        work_scale = 1;
        is_work_scale_set = true;
    }
    else {
        if (!is_work_scale_set) {
            work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
            is_work_scale_set = true;
        }
        resize(full_img, img, cv::Size(), work_scale, work_scale);
    }
    if (!is_seam_scale_set) {
        seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
        seam_work_aspect = seam_scale / work_scale;
        is_seam_scale_set = true;
    }
    
    (*finder)( img, features_global[i] );
    features_global[i].img_idx = i;
    
    BPLog("Features in image: #%d %lu", i + 1, features_global[i].keypoints.size());
    
    resize(full_img, img, cv::Size(), seam_scale, seam_scale);
    
    img.convertTo(images_global[i], CV_32F);

    finder->collectGarbage();
    full_img.release();
    img.release();
    if (int_params.back() == 1) {
        std::cout << "LAST";
    }
    addTaskToThread1(on_features_found_callback, "on_features_found", int_params, string_params);
}

static int addImagePath(const char* path, int index, int isLastImage) {
    if (isProcessing) {
        BPLog("Is Processing - can't add image %i, %i\n", index, isLastImage);
        return -1;
    }
    else if (isLastImage) {
        isProcessing = true;
    }
    
    BPLog("Adding image # %d : %s : isLastImage %d \n", index, path, isLastImage);
    
    img_names.push_back(path);
    
    cameras_global[index].R = Mat();
    
    list < string > string_params;
    list<int> int_params;
    string_params.push_back(path);
    int_params.push_back(index);
    int_params.push_back(isLastImage);
    

    addTaskToThread1(feature_finding_task, "FEATURE FIND TASK CALLED_1", int_params, string_params);
    
    return 0;
}

static int addImagePathWithAttitude(const char* path, int index, int isLastImage, double w, double x, double y, double z) {
    if (isProcessing) {
        BPLog("Is Processing - can't add image %i, %i\n", index, isLastImage);
        return -1;
    }
    else if (isLastImage) {
        isProcessing = true;
    }
    
    BPLog("Adding image with attitude # %d : %s : isLastImage: %d\n", index, path, isLastImage);
    
    img_names.push_back(path);
    
    rawAttitude << index << ",\t" << w << ",\t" << x << ",\t" << y << ",\t" << z << '\n';
    
    {   // process w,x,y,z device orientation quaternion
        double lsq = x*x + y*y + z*z + w*w;
        if (lsq == 0) {
            //cameras_global[index].R = Mat::eye(3, 3, CV_32F);
            cameras_global[index].R = Mat::Mat();
        }
        else {
            double divmult = 2.0 / lsq;
            double xx = divmult * x * x;
            double yy = divmult * y * y;
            double zz = divmult * z * z;
            
            double wx = divmult * w * x;
            double wy = divmult * w * y;
            double wz = divmult * w * z;
            double xy = divmult * x * y;
            double xz = divmult * x * z;
            double yz = divmult * y * z;
            
            cameras_global[index].R = (Mat_<float>(3, 3) << ( 1 - yy - zz ),    -( xy - wz ),   -( xz + wy ),
                                       ( xy + wz ),        -( 1 -xx -zz ), -( yz - wx ),
                                       ( xz - wy ),        -( yz + wx ),   -( 1 -xx -yy )   );
            
            // Use higher confidence threshold for adjuster if camera attitudes are already known.
            conf_thresh = 0.8f;
            
        }
    }
    
    
    //    Mat base = (Mat_<float>(3, 3) << m11, m12, m13, m21, m22, m23, m31, m32, m33);
    //    Mat mult = (Mat_<float>(3, 3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);
    //    cameras_global[index].R = base * mult;
    // Need to swizzle the matrix elements so the matrix is suitable for the initial camera estimates
    // equivalent to right-multiplying by (1, 0, 0; 0, 0, -1; 0, 1, 0);
    //cameras_global[index].R = (Mat_<float>(3, 3) << m11, m13, -m12, m21, m23, -m22, m31, m33, -m32);
    
    list < string > string_params;
    list<int> int_params;
    string_params.push_back(path);
    int_params.push_back(index);
    int_params.push_back(isLastImage);
    
    addTaskToThread1(feature_finding_task, "FEATURE FIND TASK CALLED_1", int_params, string_params);
    
    return 0;
}

#endif
