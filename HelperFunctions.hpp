//
//  HelperFunctions.hpp
//  Stitcher
//
//  Created by Chris Radford on 2017-08-02.
//  Copyright © 2017 Chris Radford. All rights reserved.
//

#ifndef HelperFunctions_h
#define HelperFunctions_h

#include "StitchingFunctions.hpp"
#include <iostream>
#include <locale>
#include <tuple>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib.hpp>
#include "opencv2/stitching/detail/blenders.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//Decleration of stitching Functions as defined in HelperFunctions.cpp
vector<Point2f> fourCorners(Mat);
tuple<float, float, float, float> minMaxCorners(vector<Point2f>);
vector<DMatch> findMatches(Mat desc1, vector< DMatch > matches);
Mat border(Mat mask);
#endif /* HelperFunctions_h */
