//
//  StitchingFunctions.cpp
//  Stitcher Project
//
//  Created by Chris Radford on 2017-06-02.
//  Copyright © 2017 Chris Radford. All rights reserved.
//


//- Definided functions as declared in StitngiFunctions.hpp
#include "StitchingFunctions.hpp"
#include "HelperFunctions.hpp"
/**-----------------------------------------
 FUNCTION:	Main stitching function that creates stiched iamge
 ARGUMENTS:  First and second image to be stitched, master lists of keypoints and vectors, desired feature detector, desired feature extractor, desired keypoint matcher, string wether or not you wish to display the matches after each iteration
 - stitch(img1,img2,keypoints, descriptors, featureDetection,featureExtractor,keypointsMatcher,showMatches);
 RETURNS:    Returns resulting image and master lists of keypoints and descriptors
 - {croppedResult,keypoints_updated,descriptors_updated}
 NOTES:
 ----------------------------------------------------**/
tuple<Mat,vector<KeyPoint>,Mat> stitch(Mat img1,Mat img2 ,vector<KeyPoint> keypoints, Mat descriptors, String featureDetection,String featureExtractor,String keypointsMatcher,String showMatches){
    
    Mat desc, desc1, desc2, homography, result, croppedResult,descriptors_updated, image1Updated, image2Updated, mask1, mask2;
    vector<KeyPoint> keypoints_updated, kp1, kp2;
    vector<DMatch> matches;
    //-Base Case[2]
    if (keypoints.empty()){
        //-Detect Keypoints and their descriptors
        tie(kp1,desc1) = KeyPointDescriptor(img1, featureDetection,featureExtractor);
        tie(kp2,desc2) = KeyPointDescriptor(img2, featureDetection,featureExtractor);
        
        //Find matches and calculated homography based on keypoints and descriptors
        tie(matches,homography) = matchFeatures(kp1,  desc1,kp2, desc2, keypointsMatcher);
        //draw matches if requested
        if(showMatches == "true")
            drawMatchedImages( img1, kp1, img2, kp2, matches);
        
        //stitch the images and update the keypoint and descriptors
        tie(image1Updated, image2Updated, mask1, mask2,keypoints_updated,descriptors_updated) = stitchMatches(img1, img2, homography,kp1,kp2,desc1,desc2);
        result = Blend(image1Updated, image2Updated, mask1, mask2);
        result = crop(result);
        return {result,keypoints_updated,descriptors_updated};
    }
    
    //base case[3:n]
    else{
        //Get keypoints and descriptors of new image and add to respective lists
        tie(kp2,desc2) = KeyPointDescriptor(img2, featureDetection,featureExtractor);
        
        
        //find matches and determine homography
        tie(matches,homography) = matchFeatures(keypoints,descriptors,kp2,desc2, keypointsMatcher);
        //draw matches if requested
        if(showMatches == "true")
            drawMatchedImages( img1, keypoints, img2, kp2, matches);
    
        //stitch the images and update the keypoint and descriptors
        tie(image1Updated, image2Updated, mask1, mask2,keypoints_updated,descriptors_updated) = stitchMatches(img1, img2, homography,keypoints,kp2,descriptors,desc2);
        result = Blend(image1Updated, image2Updated, mask1, mask2);
        result = crop(result);
        return {result,keypoints_updated,descriptors_updated};
    }
}

/**------------------------------------------
 FUNCTION:	Detects and describes keypoints and their corresponding descriptor
 ARGUMENTS:  image of any type, string of feature detector, string of feature extractor
 - KeyPointDescriptor(image1, featureDetection,featureExtractor)
 RETURNS:    Keypoints and descriptors in tuple form
 - (kp,desc)
 NOTES:      Can modify to use any type of descriptor sush as SIFT, ORB, etc.
 ----------------------------------------------------**/
tuple<vector<KeyPoint>,Mat> KeyPointDescriptor(Mat image, String featureDetection,String featureExtraction){
    cvtColor(image,image,COLOR_BGR2GRAY);
    //int minHessian = 700;
    vector<KeyPoint> keypoints;
    Mat descriptor;
    //-Section for Algoithms that can both detect and extract keypoints and their descriptors
    //-- SURF Detector & extractor
    if(featureDetection == "surf"){
        //cout << "Detect the keypoints using SURF Detector (minHEssian of 700)";
        Ptr<SURF> detector = SURF::create();
        if(featureExtraction == "surf"){
            //cout << " and using SURF extractor" << endl;
            detector->detectAndCompute(image,noArray(),keypoints,descriptor);
            return {keypoints,descriptor};
        }
        else
            detector->detect(image,keypoints);
    }
    //-- SIFT Detector & extractor
    else if(featureDetection == "sift" ){
        //cout << "Detect the keypoints using SIFT Detector (minHEssian not set)";
        Ptr<SIFT> detector = SIFT::create();
        if(featureExtraction == "sift"){
            //cout << " and using SIFT extractor" << endl;
            detector->detectAndCompute(image,noArray(),keypoints,descriptor);
            return {keypoints,descriptor};
        }
        else
            detector->detect(image,keypoints);
    }
    //-- SURF Detector & extractor
    else if(featureDetection == "orb"){
        //cout << "Detect the keypoints using ORB Detector (minHEssian of 700)" << endl;
        Ptr<ORB> detector = ORB::create();
        if(featureExtraction == "orb"){
            //cout << " and using ORB extractor" << endl;
            detector->detectAndCompute(image,noArray(),keypoints,descriptor);
            return {keypoints,descriptor};
        }
        else
            detector->detect(image,keypoints);
    }
    //-- BRISK Detector & Extractor
    else if(featureDetection == "brisk" ){
        //cout << "Detect the keypoints using BRISK Detector(no minHEssian)" << endl;
        Ptr<BRISK> detector = BRISK::create();
        if(featureExtraction == "brisk"){
            //cout << " and using BRISK extractor" << endl;
            detector->detectAndCompute(image,noArray(),keypoints,descriptor);
            return {keypoints,descriptor};
        }
        else
            detector->detect(image,keypoints);
    }
    
    //-Section for Detectors Only algorithms
    else if(featureDetection == "fast"){
        //cout << "Detect the keypoints using FAST Detector (minHEssian of 700)";
        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
        detector->detect(image, keypoints);
    }
    //-- Step 1: Detect the keypoints using STAR Detector and BREIF Extractor
    else if(featureDetection == "star"){
        //cout << "Detect the keypoints using STAR Detector (minHEssian of 30)";
        Ptr<StarDetector> detector = StarDetector::create();
        detector->detect(image, keypoints);
    }
    
    
    //-Section for Extractors/Descirptors Only
    if(featureExtraction == "brief"){
        //cout << " and using BRIEF extractor" << endl;
        Ptr<BriefDescriptorExtractor> featureExtractor = BriefDescriptorExtractor::create();
        featureExtractor->compute(image, keypoints, descriptor);
    }
    else if (featureExtraction == "freak"){
        //cout << " and using FREAK extractor" << endl;
        Ptr<DescriptorExtractor> featureExtractor = FREAK::create();
        featureExtractor->compute(image, keypoints, descriptor);
    }
    
    return {keypoints,descriptor};
}


/**-----------------------------------------
 FUNCTION:	Generates matches between images based on their keypoints
 and descriptors and computes the homography between images.
 ARGUMENTS:  keypoints and descriptors of image one and two respectively. String of meater type
 - matchFeatures(kp1, desc1, kp2, desc2, keypointsMatcher);
 RETURNS:    Returns the found matches and homography matrix in tuple form
 - (matches,homography)
 NOTES:      Function can be modified to use any Matcher found in OpenCV 3.X
 ----------------------------------------------------**/
tuple<vector< DMatch >,Mat> matchFeatures(vector<KeyPoint> kp1, Mat desc1, vector<KeyPoint> kp2, Mat desc2, String keypointsMatcher){
    vector< DMatch > matches, good_matches;
    vector<vector< DMatch >> knnMatches;
    Mat homography;
    vector<Point2f> master, addition;

    
    //-Run Brute Force based matcher. Will create vector (vector<DMatch>)
    if(keypointsMatcher == "bruteForce"){
        //cout << "Matches found using brute force algorithm" << endl;
        BFMatcher matcher;
        matcher.match( desc1, desc2, matches);
        good_matches = findMatches(desc1, matches);
    }
    //-Run Flannn based matcher. Will create vector (vector<DMatch>)
    else if( keypointsMatcher == "flann" ){
        //cout << "Matches found using Flann algorithm" << endl;
        if(desc1.type()!=CV_32F)
            desc1.convertTo(desc1, CV_32F);
        if(desc2.type()!=CV_32F)
            desc2.convertTo(desc2, CV_32F);
        
        FlannBasedMatcher matcher;
        matcher.match( desc1, desc2, matches);
        good_matches = findMatches(desc1, matches);
    }
    
     //-Run Knn Flann.KnnMatch based matcher. Will create a vector of vectors (vector<vector< DMatch >>)
     else if (keypointsMatcher == "knn"){
         //cout << "Matches found using Flann w/ KNN algorithm" << endl;
         if(desc1.type()!=CV_32F) {
             desc1.convertTo(desc1, CV_32F);
         }
         if(desc2.type()!=CV_32F) {
             desc2.convertTo(desc2, CV_32F);
         }
     
         FlannBasedMatcher matcher;
         matcher.knnMatch(desc1, desc2, knnMatches, 50);
         // Since knn retains distance use lowes ratio.
         for (int i = 0; i < knnMatches.size(); ++i){
             const float ratio = 0.6; // As in Lowe's paper;
             if (knnMatches[i][0].distance < ratio * knnMatches[i][1].distance){
                 good_matches.push_back(knnMatches[i][0]);
             }
         }
     }
    
    else{
        cout << "You did not proivde a listed matching algorithm (flann, bruteForce, knn) aborting" << endl;
        return{matches,homography};
    }

    
    
    
    
    //-- Get the keypoints from the good matches
    for( int i = 0; i < good_matches.size(); i++ )
    {
        master.push_back( kp1[ good_matches[i].queryIdx ].pt );
        addition.push_back( kp2[ good_matches[i].trainIdx ].pt );
    }
    
    //-- truncate to only keep the first 20 matdhes
    if (good_matches.size() > 20){
        good_matches.resize(20);
    }
    
    
    //-- Calculate homography
    if(good_matches.size() >=4){
        vector<unsigned char> match_mask;
        homography = findHomography( master, addition, match_mask, RANSAC, 4.0 );
        return{good_matches,homography};
    }
    else{
        cout << "Not enough matches found" << endl;
        return{good_matches,homography};
    }
}

/**-----------------------------------------
 FUNCTION:	 Draw matches found between two images
 ARGUMENTS:  Both images, their respective keypoint and found matches
 - drawMatchedImages(image1, kp1, image2, kp2, matches)
 RETURNS:    Void
 NOTES:
 ----------------------------------------------------**/
void drawMatchedImages(Mat image1, vector<KeyPoint> kp1, Mat image2, vector<KeyPoint> kp2, std::vector< DMatch > matches){
    Mat imageMatches;
    drawMatches( image1, kp1, image2, kp2, matches, imageMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    resize(imageMatches,imageMatches,Size(1200,900));
    imshow("Matches", imageMatches);
    waitKey(0);
    destroyAllWindows();
}

/**-----------------------------------------
 FUNCTION:	Stitches images based on provided homography
 ARGUMENTS:  both images and a homography matrix
 - stitch(image1, image2, Homography);
 RETURNS:    Returns resulting image
 - (result)
 NOTES:
 ----------------------------------------------------**/
tuple<Mat, Mat, Mat, Mat,vector<KeyPoint>, Mat>  stitchMatches(Mat image1,Mat image2, Mat homography, vector<KeyPoint> kp1, vector<KeyPoint> kp2 , Mat desc1, Mat desc2){
    Mat result, descriptors_updated;
    vector<Point2f> fourPointImage1, fourPointImage2, image1dst, image2dst;
    vector<KeyPoint> keypoints_updated;
    float min_x, min_y, max_x, max_y, result_x, result_y, img1_min_x, img1_min_y, img2_min_x, img2_min_y,img1_max_x, img1_max_y, img2_max_x, img2_max_y, result_min_x, result_max_x, result_min_y, result_max_y;
    
    //get four corners of image MATs
    fourPointImage1 = fourCorners(image1);
    fourPointImage2 = fourCorners(image2);
    
    //get the min and max corners
    tie(min_x, min_y, max_x, max_y) = minMaxCorners(fourPointImage1);
     
    //- Htr use to map image one to result in line with the already warped image two
    Mat Htr = Mat::eye(3,3,CV_64F);
    if (min_x < 0){
        max_x = image2.size().width - min_x;
        Htr.at<double>(0,2)= -min_x;
    }
    if (min_y < 0){
        max_y = image2.size().height - min_y;
        Htr.at<double>(1,2)= -min_y;
    }
    
    //- Need to create a padded blank image to accomodate stitched images.
    //- Must first determine where the translations of the two images will end up to determine if shift is needed and how much to pad blank image
    perspectiveTransform(fourPointImage1, image1dst, Htr*homography);
    perspectiveTransform(fourPointImage2, image2dst, Htr);
  
    //- Now determine what is out of bounds
    //New coordinates for image 1
    tie(img1_min_x, img1_min_y,img1_max_x,img1_max_y) = minMaxCorners(image1dst);
    //cout << "Image 1: Min x " << img1_min_x << " Min y " << img1_min_y << " Max x " << img1_max_x << " Max y " << img1_max_y << endl;
    
    //New coordinates for iamge 2
    tie(img2_min_x, img2_min_y,img2_max_x,img2_max_y) = minMaxCorners(image2dst);
    //cout << "Image 2: Min x " << img2_min_x << " Min y " << img2_min_y << " Max x " << img2_max_x << " Max y " << img2_max_y << endl;
    
    //determine bounding area for resulting images
    result_min_x = min(img1_min_x, img2_min_x);
    result_max_x = max(img1_max_x, img2_max_x);
    result_min_y = min(img1_min_y, img2_min_y);
    result_max_y = max(img1_max_y, img2_max_y);
    
    //cout << "Result Image: Min x " << result_min_x << " Min y " << result_min_y << " Max x " << result_max_x << " Max y " << result_max_y << endl;
    
    //Determine size of padded result (blank image) to be used for stitching
    result_x = (abs(floor(result_min_x)) + abs(ceil(result_max_x)));
    result_y = (abs(floor(result_min_y)) + abs(ceil(result_max_y)));
    
    result = Mat(Size(result_x,result_y), CV_32FC1,Scalar(0,0,0));
    
    
    //-Move the images to the postiive by creating a matrix to represent the translation
    int anchor_x, anchor_y = 0;
    Mat translation = Mat::eye(3,3,CV_64F);
    if(result_min_x < 0){
        anchor_x = (int) floor(result_min_x);
        translation.at<double>(0,2) -= anchor_x;
    }
    if(result_min_y < 0){
        anchor_y = (int) floor(result_min_y);
        translation.at<double>(1,2) -= anchor_y;
    }

    //cout << translation << endl;
    
    //Warp each image accordingly to the new padded result image
    //warpPerspective(image1, result, (translation*homography), result.size(), INTER_LINEAR, BORDER_CONSTANT,(0));
    //warpPerspective(image2, result, translation, result.size(), INTER_LINEAR, BORDER_TRANSPARENT,   (0));
    

    //----------- Section for blending function
    //Mask of image to be combined so you can get resulting mask
    Mat mask1(image1.size(), CV_8UC1, Scalar::all(255));
    Mat mask2(image2.size(), CV_8UC1, Scalar::all(255));
    Mat image1Updated, image2Updated;
    
    
    //Warp the masks and the images to their new posistions so their are of all the same  size to be overlayed and blended
    warpPerspective(image1, image1Updated, (translation*homography), result.size(), INTER_LINEAR, BORDER_CONSTANT,(0));
    warpPerspective(image2, image2Updated, translation, result.size(), INTER_LINEAR, BORDER_CONSTANT,   (0));
    warpPerspective(mask1, mask1, (translation*homography), result.size(), INTER_LINEAR, BORDER_CONSTANT,(0));
    warpPerspective(mask2, mask2, translation, result.size(), INTER_LINEAR, BORDER_CONSTANT,   (0));
    
    //-- Variables to hold the keypoints at the respective stages
    vector<Point2f> kp1Local,kp2Local;
    vector<KeyPoint> kp1updated, kp2updated;
    
    //Localize the keypoints to allow for perspective change
    KeyPoint::convert(kp1, kp1Local);
    KeyPoint::convert(kp2, kp2Local);
    
    //perform persepctive transform on the keypoints of type vector<point2f>
    perspectiveTransform(kp1Local, kp1Local, (translation*homography));
    perspectiveTransform(kp2Local, kp2Local, (translation));
    
    //convert keypoints back to type vector<keypoint>
    for( size_t i = 0; i < kp1Local.size(); i++ ) {
        kp1updated.push_back(KeyPoint(kp1Local[i], 1.f));
    }
    for( size_t i = 0; i < kp2Local.size(); i++ ) {
        kp2updated.push_back(KeyPoint(kp2Local[i], 1.f));
    }
    
    //Add to master of list of keypoints to be passed along during next iteration of image
    keypoints_updated.reserve(kp1updated.size() + kp2updated.size());
    keypoints_updated.insert(keypoints_updated.end(),kp1updated.begin(),kp1updated.end());
    keypoints_updated.insert(keypoints_updated.end(),kp2updated.begin(),kp2updated.end());
    
    //create a new Mat including the descriports from desc1 and desc2
    descriptors_updated.push_back(desc1);
    descriptors_updated.push_back(desc2);
    
    return {image1Updated, image2Updated, mask1, mask2, keypoints_updated, descriptors_updated};
}
/**-----------------------------------------
 FUNCTION:	Blends the two images passed to it
 ARGUMENTS:  both images (already warped) and their corresponding masks.]
 - Blend(image1Updated, image2Updated, mask1, mask2);
 RETURNS:    Returns resulting blended image
 - (merged)
 NOTES:
 ----------------------------------------------------**/
Mat Blend(Mat image1Updated, Mat image2Updated, Mat mask1, Mat mask2){
    Mat bothMasks, noMask, rawAlpha, border1, border2, dist1, dist2;
    bothMasks = mask1 | mask2;
    // invert mask to get borders
    noMask = 255-bothMasks;

    
    // create an image with equal alpha values:
    rawAlpha = cv::Mat(noMask.rows, noMask.cols, CV_32FC1);
    rawAlpha = 1.0f;
    
    // invert the border, so that border values are 0 ... this is needed for the distance transform
    border1 = 255-border(mask1);
    border2 = 255-border(mask2);

    
    // compute the distance to the object center
    distanceTransform(border1,dist1,CV_DIST_L2, 3);
    
    // scale distances to values between 0 and 1
    double min, max; cv::Point minLoc, maxLoc;
    
    // find min/max vals
    minMaxLoc(dist1,&min,&max, &minLoc, &maxLoc, mask1&(dist1>0));  // edited: find min values > 0
    dist1 = dist1* 1.0/max; // values between 0 and 1 since min val should alwaysbe 0
    
    // same for the 2nd image
    distanceTransform(border2,dist2,CV_DIST_L2, 3);
    minMaxLoc(dist2,&min,&max, &minLoc, &maxLoc, mask2&(dist2>0));  // edited: find min values > 0
    dist2 = dist2*1.0/max;  // values between 0 and 1
    
    
    
    // mask the distance values to reduce information to masked regions
    Mat dist1Masked;
    rawAlpha.copyTo(dist1Masked,noMask);    // edited: where no mask is set, blend with equal values
    dist1.copyTo(dist1Masked,mask1);
    rawAlpha.copyTo(dist1Masked,mask1&(255-mask2)); //edited
    
    Mat dist2Masked;
    rawAlpha.copyTo(dist2Masked,noMask);    // edited: where no mask is set, blend with equal values
    dist2.copyTo(dist2Masked,mask2);
    rawAlpha.copyTo(dist2Masked,mask2&(255-mask1)); //edited
    
    // dist1Masked and dist2Masked now hold the "quality" of the pixel of the image, so the higher the value, the more of that pixels information should be kept after blending
    // problem: these quality weights don't build a linear combination yet
    
    // you want a linear combination of both image's pixel values, so at the end you have to divide by the sum of both weights
    cv::Mat blendMaskSum = dist1Masked+dist2Masked;
    
    //convert the images to float to multiply with the weight
    Mat im1Float;
    image1Updated.convertTo(im1Float,dist1Masked.type());
    
    vector<Mat> channels1;
    split(im1Float,channels1);
    // multiply pixel value with the quality weights for image 1
    Mat im1AlphaB = dist1Masked.mul(channels1[0]);
    Mat im1AlphaG = dist1Masked.mul(channels1[1]);
    Mat im1AlphaR = dist1Masked.mul(channels1[2]);
    
    vector<Mat> alpha1;
    alpha1.push_back(im1AlphaB);
    alpha1.push_back(im1AlphaG);
    alpha1.push_back(im1AlphaR);
    Mat im1Alpha;
    merge(alpha1,im1Alpha);
    
    Mat im2Float;
    image2Updated.convertTo(im2Float,dist2Masked.type());
    
    vector<Mat> channels2;
    split(im2Float,channels2);
    // multiply pixel value with the quality weights for image 2
    Mat im2AlphaB = dist2Masked.mul(channels2[0]);
    Mat im2AlphaG = dist2Masked.mul(channels2[1]);
    Mat im2AlphaR = dist2Masked.mul(channels2[2]);
    
    vector<Mat> alpha2;
    alpha2.push_back(im2AlphaB);
    alpha2.push_back(im2AlphaG);
    alpha2.push_back(im2AlphaR);
    Mat im2Alpha;
    merge(alpha2,im2Alpha);
    
    // now sum both weighted images and divide by the sum of the weights (linear combination)
    Mat imBlendedB = (im1AlphaB + im2AlphaB)/blendMaskSum;
    Mat imBlendedG = (im1AlphaG + im2AlphaG)/blendMaskSum;
    Mat imBlendedR = (im1AlphaR + im2AlphaR)/blendMaskSum;
    vector<Mat> channelsBlended;
    channelsBlended.push_back(imBlendedB);
    channelsBlended.push_back(imBlendedG);
    channelsBlended.push_back(imBlendedR);
    
    // merge back to 3 channel image
    Mat merged;
    merge(channelsBlended,merged);
    
    // convert to 8UC3
    merged.convertTo(merged,CV_8UC3);
    return(merged);
}

/**-----------------------------------------
 FUNCTION:	Crops out the black portions of an image
 ARGUMENTS:  An image
 - crop(image);
 RETURNS:    Returns resulting croppedimage
 - (cropped)
 NOTES:
 ----------------------------------------------------**/
Mat crop(Mat image){
    Mat cropped, grayed, thresh, transparent, result;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    cvtColor(image, grayed, CV_BGR2GRAY);
    threshold( grayed, thresh, 1, 255,THRESH_BINARY);
    findContours( thresh, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    vector<int> x,y;
    for(int i=0; i < contours.size(); i++){
        for(int j = 0; j < contours.at(i).size();j++){
            x.push_back(contours[i][j].x);
            y.push_back(contours[i][j].y);
        }
    }

    auto xVals = std::minmax_element(x.begin(), x.end());
    auto yVals = std::minmax_element(y.begin(), y.end());
    
    Rect rect (*xVals.first,*yVals.first,(*xVals.second)-(*xVals.first),(*yVals.second)-(*yVals.first));
    cropped = image(rect);
    return cropped;
}
