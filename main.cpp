//
//  main.cpp
//  Stitcher
//
//  Created by Chris Radford on 2017-05-22.
//  Copyright © 2017 Chris Radford. All rights reserved.
//


//Header linking files
#include "StitchingFunctions.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <locale>
#include <tuple>
#include <list>

using namespace boost::filesystem;

/**------------------------------------------
 FUNCTION:	Main stitch function that stores all necessary and found data to stitch images
 ARGUMENTS:  a set of two images of any type and Bool stating showMatches condition and string of all specifications.
 This function will be called from another program and therefore require constructor
 - stitcher(image1, image2, featureDetection, featureExtractor, keypointsMatcher)
 RETURNS:    Returns resulting image that has been stitched
 - return Result
 ----------------------------------------------------**/
int main(int argc, char *argv[])
{
    path p = argv[1];
    String featureDetection = argv[2];
    String featureExtractor = argv[3];
    String keypointsMatcher = argv[4];
    String showMatches = argv[5];
    vector<directory_entry> files; // Grab file names
    vector<string> imageList; //list of images in string form
    Mat masterImage; //resulting image
    Mat nextImage;
    std::vector<KeyPoint> keypoints; //List of all found keypoints
    Mat descriptors; //List of all discriptors
    
    // confirm path is a directory
    if(is_directory(p)){
        
        //grab files and put into a list of type directory_entry
        copy(directory_iterator(p), directory_iterator(), back_inserter(files));
        //convert list of type direct_entry to of type string
        for ( vector<directory_entry>::const_iterator it = files.begin(); it != files.end();  ++ it ){
            String extension = boost::filesystem::extension(*it);
            
            //make sure only taking in images and no hidden files
            if(extension != ".jpg")
                continue;
            imageList.push_back((*it).path().string());
        }
        //Check that the first file found is an image and not a hidden file
        //String file = imageList.front();
        //-Base Cse[0:1]
        if(imageList.size() < 2){
            cout << "Not enough images to stitch" << endl;
            return 0;
        }
        //-Base Case[2]
        else{
            vector<string>::iterator it = imageList.begin();
            //Get first iamge
            advance(it,0);
            Mat img1 = imread(*it,1);
            resize(img1,img1,Size(1200,900));
            //Get second image
            advance(it,1);
            Mat img2 = imread(*it,1);
            resize(img2,img2,Size(1200,900));
            cout << "Stitching first two images " << endl;
            tie(masterImage,keypoints,descriptors) = stitch(img1,img2,keypoints, descriptors, featureDetection,featureExtractor,keypointsMatcher,showMatches);
            if(masterImage.rows == 0){
                cout << "Aborting main" << endl;
                //return 0;
            }
            
            //-Base Case[3:n]
            if (imageList.size() > 2){
                //remove first to images as they have already been dealt with
                imageList.erase(imageList.begin());
                imageList.erase(imageList.begin());
                //Iterate throuh remaining images adding them to masterImage
                for ( vector<string>::iterator it = imageList.begin(); it != imageList.end();  ++ it ){
                    cout << "Now Stitching image: " << *it << endl;
                    //nextImage = imread((*it),1);
                    //long e1 = getTickCount();
                    resize(imread( (*it),1),nextImage,Size(1200,900));
                    //long e2 = getTickCount();
                    //float time = (e2-e1)/getTickFrequency();
                    //cout << time << endl;
                    tie(masterImage,keypoints,descriptors) = stitch(masterImage,nextImage,keypoints, descriptors, featureDetection,featureExtractor,keypointsMatcher,showMatches);
                    if(masterImage.rows == 0){
                        cout << "Aborting main" << endl;
                        return 0;
                    }
                }
            }
        }
        cout << "Done" << endl;
        //resize(masterImage,masterImage,Size(1200,900));
        imwrite("/Users/chrisradford/Documents/School/Masters/Research/result.jpg", masterImage);
        imshow("Result",masterImage);
        waitKey(0);
        destroyAllWindows();
        return 0;
    }
}

