//
//  StitchingFunctions.hpp
//  Stitcher
//
//  Created by Chris Radford on 2017-06-19.
//  Copyright © 2017 Chris Radford. All rights reserved.
//

#ifndef StitchingFunctions_hpp
#define StitchingFunctions_hpp

#include "StitchingFunctions.hpp"
#include <iostream>
#include <locale>
#include <tuple>
#include <cmath>
#include <opencv2/core.hpp>
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

//Decleration of stitching Functions as defined in StictingFunctions.cpp
tuple<vector<KeyPoint>,Mat> KeyPointDescriptor(Mat , String ,String );
tuple<vector< DMatch >,Mat> matchFeatures(vector<KeyPoint>, Mat, vector<KeyPoint>, Mat, String);
void drawMatchedImages(Mat, vector<KeyPoint>, Mat, vector<KeyPoint>, vector<DMatch>);
tuple<Mat, Mat, Mat, Mat, vector<KeyPoint>, Mat> stitchMatches(Mat, Mat, Mat, vector<KeyPoint>, vector<KeyPoint>, Mat, Mat );
tuple<Mat,vector<KeyPoint>,Mat> stitch(Mat,Mat,vector<KeyPoint> keypoints, Mat descriptors, String,String,String,String);
Mat Blend(Mat image1Updated, Mat image2Updated, Mat mask1, Mat mask2);
Mat crop(Mat image);

#endif /* StitchingFunctions_hpp */
