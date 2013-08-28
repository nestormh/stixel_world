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

#include "fundamentalmatrixestimator.h"

#include "utils.h"

using namespace stixel_world;

bool FundamentalMatrixEstimator::findF(const cv::Mat& imgLt0, const cv::Mat& imgRt0, 
                                       const cv::Mat& imgLt1, const cv::Mat& imgRt1, 
                                       cv::Mat& FL, cv::Mat& FR, vector < vector < cv::Point2f > > & finalCorrespondences,
                                       const double & cornerThresh)
{
    vector < vector < cv::Point2f > > initialPoints(5), points;
    findInitialPoints(imgLt0, initialPoints[0], cornerThresh);
    
    findPairCorrespondences(imgLt0, imgRt0, initialPoints[0], initialPoints[1]);
    findPairCorrespondences(imgRt0, imgRt1, initialPoints[1], initialPoints[2]);
    findPairCorrespondences(imgRt1, imgLt1, initialPoints[2], initialPoints[3]);
    findPairCorrespondences(imgLt1, imgLt0, initialPoints[3], initialPoints[4]);
    
    cleanCorrespondences(initialPoints, points);
    
    select8Points(points, finalCorrespondences);
    
    FL = cv::findFundamentalMat(finalCorrespondences[0], finalCorrespondences[3], CV_FM_8POINT);
    FR = cv::findFundamentalMat(finalCorrespondences[1], finalCorrespondences[2], CV_FM_8POINT);
    
    if ((cv::countNonZero(FL) == 0) || (cv::countNonZero(FR) == 0) ||
        (! cv::checkRange(FL)) || (!cv::checkRange(FR)))
        return false;
    
    visualize(imgLt0, imgRt0, imgLt1, imgRt1, initialPoints[0], points, finalCorrespondences);
    
    return true;
}

inline void FundamentalMatrixEstimator::findInitialPoints(const cv::Mat& img, vector< cv::Point2f >& points, const double & cornerThresh)
{
    vector<cv::KeyPoint> keypoints;
    cv::FastFeatureDetector fastDetector(cornerThresh);
    fastDetector.detect(img, keypoints);
    
    if (keypoints.size() == 0)
        return;
    
    points = vector<cv::Point2f>(keypoints.size());
    {
        uint32_t idx = 0;
        for (vector<cv::KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); it++, idx++) {
            points[idx] = it->pt;
        }
    }    
}

inline void FundamentalMatrixEstimator::findPairCorrespondences(const cv::Mat& img1, const cv::Mat& img2, const vector< cv::Point2f >& points1, vector< cv::Point2f >& points2)
{
    // Optical flow
    vector<uint8_t> status, statusB;
    vector<float_t> error, errorB;
    
    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, error, cv::Size(3, 3), 3);
}

inline void FundamentalMatrixEstimator::cleanCorrespondences(const vector< vector< cv::Point2f > >& initialCorrespondences, 
                                                             vector< vector< cv::Point2f > >& finalCorrespondences)
{
    finalCorrespondences.resize(4);
    for (uint32_t i = 0; i < 4; i++) {
        finalCorrespondences[i].reserve(initialCorrespondences[0].size());
    }
    
    for (uint32_t i = 0; i < initialCorrespondences[0].size(); i++) {
        const cv::Point2f & p1 = initialCorrespondences[0][i];
        const cv::Point2f & p2 = initialCorrespondences[4][i];
        
        float dist = cv::norm(p1 - p2);
        
        if (dist < 1.0) {
            finalCorrespondences[0].push_back(p1);
            finalCorrespondences[1].push_back(initialCorrespondences[1][i]);
            finalCorrespondences[2].push_back(initialCorrespondences[2][i]);
            finalCorrespondences[3].push_back(initialCorrespondences[3][i]);
        }
    }
}

inline void FundamentalMatrixEstimator::select8Points(vector< vector< cv::Point2f > >& correspondences, 
                                                      vector< vector< cv::Point2f > >& finalCorrespondences)
{
    const vector< cv::Point2f > & points = correspondences[0];
    vector< int32_t > hull;
    
    
    finalCorrespondences.resize(correspondences.size());
    for (uint32_t i = 0; i < finalCorrespondences.size(); i++)
        finalCorrespondences[i].resize(8);
    
    cv::Mat hullM;
    cv::convexHull(points, hull, false, false);
    
    if (hull.size() >= 8) { 
        double idx2 = 0.0;
        for (uint32_t idx1 = 0/*, idx2 = 0*/; idx1 < 8; idx1++, idx2 += (double)hull.size() / 8.0) {
            for (uint32_t i = 0; i < finalCorrespondences.size(); i++) {
                finalCorrespondences[i][idx1] = correspondences[i][hull[(uint32_t)idx2]];
            }
        }
    } else {
        vector<bool> used(points.size());
        for (uint32_t i = 0; i < hull.size(); i++) {
            used[hull[i]] = true;
            for (uint32_t j = 0; j < finalCorrespondences.size(); j++)
                finalCorrespondences[j][i] = correspondences[j][hull[i]];
        }
        for (uint32_t i = hull.size(); i < 8; i++) {
            uint32_t idx = rand() % points.size();
            while (used[idx]) {
                idx++;
            }
            used[idx] = true;
            for (uint32_t j = 0; j < finalCorrespondences.size(); j++)
                finalCorrespondences[j][i] = correspondences[j][idx];
        }
    }
}

void FundamentalMatrixEstimator::visualize(const cv::Mat & imgLt0, const cv::Mat & imgRt0, const cv::Mat & imgLt1, const cv::Mat & imgRt1, 
                                           const vector< cv::Point2f > & initialPoints, 
                                           const vector< vector< cv::Point2f > > & correspondences, 
                                           const vector< vector< cv::Point2f > > & finalCorrespondences)
{
    cv::Mat Lt0, Lt1, Rt0, Rt1;
    
    imgLt0.copyTo(Lt0);
    imgRt0.copyTo(Lt1);
    imgLt1.copyTo(Rt0);
    imgRt1.copyTo(Rt1);
    
    for (uint32_t i = 0; i < finalCorrespondences[0].size(); i++) {
        cv::circle(Rt0, finalCorrespondences[0][i], 5, cv::Scalar::all(255), 2);
        cv::circle(Lt1, finalCorrespondences[1][i], 5, cv::Scalar::all(255), 2);
        cv::circle(Rt1, finalCorrespondences[2][i], 5, cv::Scalar::all(255), 2);
        cv::circle(Lt0, finalCorrespondences[3][i], 5, cv::Scalar::all(255), 2);
    }
    
    for (uint32_t i = 0; i < correspondences[0].size(); i++) {
        cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        
        cv::circle(Rt0, correspondences[0][i], 2, color, -1);
        cv::circle(Lt1, correspondences[1][i], 2, color, -1);
        cv::circle(Rt1, correspondences[2][i], 2, color, -1);
        cv::circle(Lt0, correspondences[3][i], 2, color, -1);
    }
    
    
    cv::Mat scaledLt0(300, 400, CV_8UC3);
    cv::Mat scaledRt0(300, 400, CV_8UC3); 
    cv::Mat scaledLt1(300, 400, CV_8UC3); 
    cv::Mat scaledRt1(300, 400, CV_8UC3);

    cv::Mat finalView(600, 800, CV_8UC3);
    
    cv::resize(Lt0, scaledLt0, scaledLt0.size());
    cv::resize(Lt1, scaledRt0, scaledLt0.size());
    cv::resize(Rt0, scaledLt1, scaledLt0.size());
    cv::resize(Rt1, scaledRt1, scaledLt0.size());
    
    scaledLt0.copyTo(finalView(cv::Rect(0, 0, 400, 300)));
    scaledRt0.copyTo(finalView(cv::Rect(400, 0, 400, 300)));
    scaledLt1.copyTo(finalView(cv::Rect(0, 300, 400, 300)));
    scaledRt1.copyTo(finalView(cv::Rect(400, 300, 400, 300)));
    
    cv::imshow("finalView", finalView);

    waitForKey();
}
