/*
 *  Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef FUNDAMENTALMATRIXESTIMATOR_H
#define FUNDAMENTALMATRIXESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

namespace stixel_world {
    
#define MATCH_STEREO_PAIR 0
#define MATCH_BETWEEN_FRAMES 1

#define MAX_HORIZONTAL_DIST 1.0
#define MAX_FLOW_DIST 20.0
#define MAX_CICLE_DIST 1.0
    
class FundamentalMatrixEstimator
{

public:
    static const uint8_t METHOD_OFLOW = 0;
    static const uint8_t METHOD_SURF = 1;
    static const uint8_t METHOD_COMBINED = 2;
    
    static bool findF(const cv::Mat & imgLt0, const cv::Mat & imgRt0, 
                    const cv::Mat & imgLt1, const cv::Mat & imgRt1, 
                    cv::Mat& FL, cv::Mat& FR, vector < vector < cv::Point2f > > & finalCorrespondences,
                    const uint8_t & method = METHOD_OFLOW, const double & cornerThresh = 50);
private:
    static void findInitialPoints(const cv::Mat & img, vector<cv::Point2f> & points, const double & cornerThresh);
    static void findPairCorrespondencesOFlow(const cv::Mat & img1, const cv::Mat & img2, 
                                             vector<cv::Point2f> & points1, vector<cv::Point2f> & points2,
                                             const int & matchingMode = MATCH_STEREO_PAIR);
    static void findPairCorrespondencesSURF(const cv::Mat & img1, const cv::Mat & img2, 
                                            vector<cv::KeyPoint> & keypoints1, vector<cv::KeyPoint> & keypoints2,
                                            const int & matchingMode = MATCH_STEREO_PAIR);
    static void cleanMatchesStereoPair(const vector <cv::KeyPoint> & keypoints1, const vector <cv::KeyPoint> & keypoints2,
                                       const vector<cv::DMatch> & matches1, vector<cv::DMatch> & matches2);
    static void cleanMatchesBetweenFrames(const vector <cv::KeyPoint> & keypoints1, const vector <cv::KeyPoint> & keypoints2,
                                          const vector<cv::DMatch> & matches1, vector<cv::DMatch> & matches2);
    
    static void cleanMatchesStereoPair(vector< cv::Point2f >& points1, vector< cv::Point2f >& points2);
    static void cleanMatchesBetweenFrames(vector< cv::Point2f >& points1, vector< cv::Point2f >& points2);
    
    static void cleanCorrespondences(const vector < vector < cv::Point2f > > & initialCorrespondences,
                                     vector < vector < cv::Point2f > > & finalCorrespondences);
    static void select8Points(vector < vector < cv::Point2f > > & correspondences, vector< vector< cv::Point2f > > & finalCorrespondences);
    static void visualize(const cv::Mat & imgLt0, const cv::Mat & imgRt0, 
                   const cv::Mat & imgLt1, const cv::Mat & imgRt1, 
                   const vector< cv::Point2f > & initialPoints, 
                   const vector< vector< cv::Point2f > > & correspondences, 
                   const vector< vector< cv::Point2f > > & finalCorrespondences);
    static void drawMatches(const cv::Mat & img1, const cv::Mat & img2, 
                     const vector<cv::Point2f> & points1, const vector<cv::Point2f> & points2);
    static double measureFundMatrixQuality(const vector<cv::Point2f> & points1, const vector<cv::Point2f> & points2, const cv::Mat & F);
};

}
#endif // FUNDAMENTALMATRIXESTIMATOR_H
