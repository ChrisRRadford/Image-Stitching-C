//
//  HelperFunctions.cpp
//  Stitcher
//
//  Created by Chris Radford on 2017-08-02.
//  Copyright © 2017 Chris Radford. All rights reserved.
//

#include "HelperFunctions.hpp"

//Get the four corners of a mat (Points order in clockwise direction for origin being top left)
vector<Point2f> fourCorners(Mat image){
    vector<Point2f> corners;
    corners.push_back(Point2f (0,0));                                       //-Top left
    corners.push_back(Point2f (image.size().width,0));                      //-Top right
    corners.push_back(Point2f (0, image.size().height));                    //-Bottom left
    corners.push_back(Point2f (image.size().width, image.size().height));   //Bottom right
    return(corners);
}

tuple<float, float, float, float> minMaxCorners(vector<Point2f> imageMatrix){
    float min_x1, min_x2, min_y1, min_y2, max_x1, max_x2, max_y1, max_y2,img_min_x, img_min_y, img_max_x, img_max_y;
    min_x1 = min(imageMatrix.at(0).x, imageMatrix.at(1).x);
    min_x2 = min(imageMatrix.at(2).x, imageMatrix.at(3).x);
    min_y1 = min(imageMatrix.at(0).y, imageMatrix.at(1).y);
    min_y2 = min(imageMatrix.at(2).y, imageMatrix.at(3).y);
    max_x1 = max(imageMatrix.at(0).x, imageMatrix.at(1).x);
    max_x2 = max(imageMatrix.at(2).x, imageMatrix.at(3).x);
    max_y1 = max(imageMatrix.at(0).y, imageMatrix.at(1).y);
    max_y2 = max(imageMatrix.at(2).y, imageMatrix.at(3).y);
    img_min_x = min(min_x1, min_x2);
    img_min_y = min(min_y1, min_y2);
    img_max_x = max(max_x1, max_x2);
    img_max_y = max(max_y1, max_y2);
    return{img_min_x,img_min_y, img_max_x, img_max_y };
}


vector<DMatch> findMatches(Mat desc1, vector< DMatch > matches){
    double max_dist = 0, min_dist = 100;
    vector< DMatch > good_matches;
    for(int i = 0; i < desc1.rows; i++ ){
        double dist = matches[i].distance;
    if(dist < min_dist)
        min_dist = dist;
    if(dist > max_dist)
        max_dist = dist;
    }

//-- Only take matches found within a certain distance
    for(int i = 0; i < desc1.rows; i++ ){
        if(matches[i].distance < 6*min_dist )
            good_matches.push_back( matches[i]);
    }
    return (good_matches);
}
//-- Border invetor for masks used in blending
Mat border(Mat mask)
{
    Mat gx, gy;
    
    
    Sobel(mask,gx,CV_32F,1,0,3);
    Sobel(mask,gy,CV_32F,0,1,3);
    
    Mat border;
    magnitude(gx,gy,border);
    
    return border > 100;
}


