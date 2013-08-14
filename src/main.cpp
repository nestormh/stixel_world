/*
    Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <iostream>
#include <stdio.h>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/concept_check.hpp>
#include "rectification.h"

using namespace std;
using namespace stixel_world;

#define BASE_PATH "/local/imaged/stixels/bahnhof"
#define IMG1_PATH "seq03-img-left"
#define FILE_STRING1 "image_%08d_0.png"
#define IMG2_PATH "seq03-img-right"
#define FILE_STRING2 "image_%08d_1.png"

#define CALIBRATION_STRING "cam%d.cal"

#define MIN_IDX 138 //120
#define MAX_IDX 999

int main(int argc, char * argv[]) {

    // Images are dedistorted
    boost::filesystem::path calibrationPath1(BASE_PATH);
    boost::filesystem::path calibrationPath2(BASE_PATH);
    
    char calibrationName[1024];
    
    sprintf(calibrationName, CALIBRATION_STRING, 1);
    calibrationPath1 /= calibrationName;
    sprintf(calibrationName, CALIBRATION_STRING, 2);
    calibrationPath2 /= calibrationName;
    
    Rectification rectificator;
    rectificator.readParamsFromFile(calibrationPath1.string(), 0, false);
    rectificator.readParamsFromFile(calibrationPath2.string(), 1, true);
    
    cv::namedWindow("img1");
    cv::namedWindow("img2");
    cv::moveWindow("img2", 700, 0);

    cv::Mat img1, img2;
    for (uint32_t i = MIN_IDX; i < MAX_IDX; i++) {
        boost::filesystem::path img1Path(BASE_PATH);
        boost::filesystem::path img2Path(BASE_PATH);
        
        char imageName[1024];
        sprintf(imageName, FILE_STRING1, i);
        img1Path /= IMG1_PATH;
        img1Path /= imageName;
        sprintf(imageName, FILE_STRING2, i);

        img2Path /= IMG2_PATH;
        img2Path /= imageName;
        
        cout << img1Path.string() << endl;
        cout << img2Path.string() << endl;
        
        cv::Mat distortedImg1 = cv::imread(img1Path.string(), 0);
        cv::Mat distortedImg2 = cv::imread(img2Path.string(), 0);
        
        rectificator.doRectification(distortedImg1, distortedImg2, img1, img2);
        
        cv::imshow("img1", img1);
        cv::imshow("img2", img2);
        
        uint8_t keycode = cv::waitKey(0);
        switch (keycode) {
            case 'q':
                exit(0);
                break;
            default:
                ;
        }
   }
  return 0;
}


