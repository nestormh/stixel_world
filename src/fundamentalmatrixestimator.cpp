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
#include <opencv2/nonfree/features2d.hpp>

using namespace stixel_world;

bool FundamentalMatrixEstimator::findF(const cv::Mat& imgLt0, const cv::Mat& imgRt0, 
                                       const cv::Mat& imgLt1, const cv::Mat& imgRt1, 
                                       cv::Mat& FL, cv::Mat& FR, vector < vector < cv::Point2f > > & finalCorrespondences,
                                       const uint8_t & method, const double & cornerThresh)
{
    vector < vector < cv::Point2f > > initialPoints(5), points;
    
    findInitialPoints(imgLt0, initialPoints[0], cornerThresh);

    findPairCorrespondencesOFlow(imgLt0, imgRt0, initialPoints[0], initialPoints[1], MATCH_STEREO_PAIR);
    findPairCorrespondencesOFlow(imgRt0, imgRt1, initialPoints[1], initialPoints[2], MATCH_BETWEEN_FRAMES);
    findPairCorrespondencesOFlow(imgRt1, imgLt1, initialPoints[2], initialPoints[3], MATCH_STEREO_PAIR);
    findPairCorrespondencesOFlow(imgLt1, imgLt0, initialPoints[3], initialPoints[4], MATCH_BETWEEN_FRAMES);
    
    cleanCorrespondences(initialPoints, points);
    
//     drawMatches(imgLt0, imgRt0, points[0], points[1]);
    
    if (points[0].size() < 8) {
        cout << "Not enough points!!! " << points[0].size() << endl;
        if (initialPoints[0].size() < 8) {
            return false;
        }
//         select8Points(initialPoints, finalCorrespondences);
        finalCorrespondences = initialPoints;
    } else {    
//         select8Points(points, finalCorrespondences);
        finalCorrespondences = points;
    }
    
    FL = cv::findFundamentalMat(finalCorrespondences[0], finalCorrespondences[3], CV_FM_LMEDS);
    FR = cv::findFundamentalMat(finalCorrespondences[1], finalCorrespondences[2], CV_FM_LMEDS);
    
//     waitForKey();
//     visualize(imgLt0, imgRt0, imgLt1, imgRt1, initialPoints[0], points, finalCorrespondences);
    
    if ((cv::countNonZero(FL) == 0) || (cv::countNonZero(FR) == 0) ||
        (! cv::checkRange(FL)) || (!cv::checkRange(FR)) ||
        (measureFundMatrixQuality(finalCorrespondences[0], finalCorrespondences[3], FL) > 25) ||
        (measureFundMatrixQuality(finalCorrespondences[0], finalCorrespondences[3], FR) > 25))
        return false;
    
//     waitForKey();
//     exit(0);
    
    return true;
}

inline 
void FundamentalMatrixEstimator::findInitialPoints(const cv::Mat& img, vector< cv::Point2f >& points, const double & cornerThresh)
{
    
    cv::Mat mask;
    cv::Canny(img, mask, 100, 200);
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
    
    vector<cv::KeyPoint> keypoints;
    cv::FastFeatureDetector fastDetector(cornerThresh);
    fastDetector.detect(img, keypoints, mask);
    
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

inline 
void FundamentalMatrixEstimator::findPairCorrespondencesOFlow(const cv::Mat& img1, const cv::Mat& img2, 
                                                              vector< cv::Point2f >& points1, vector< cv::Point2f >& points2,
                                                              const int & matchingMode)
{
    // Optical flow
    vector<uint8_t> status, statusB;
    vector<float_t> error, errorB;
    
    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, error, cv::Size(3, 3), 9);
        
//     if (matchingMode == MATCH_STEREO_PAIR) {
//         cleanMatchesStereoPair(points1, points2);
//     } else if (matchingMode == MATCH_BETWEEN_FRAMES) {
//         cleanMatchesBetweenFrames(points1, points2);
//     }
//     drawMatches(img1, img2, points1, points2);
}

inline 
void FundamentalMatrixEstimator::cleanMatchesStereoPair(vector< cv::Point2f >& points1, vector< cv::Point2f >& points2)
{
    const vector< cv::Point2f > oldPoints1 = points1;
    const vector< cv::Point2f > oldPoints2 = points2;
    
    points1.clear();
    points2.clear();
    
    points1.reserve(oldPoints1.size());
    points2.reserve(oldPoints2.size());
    
    for (vector<cv::Point2f>::const_iterator it1 = oldPoints1.begin(), it2 = oldPoints2.begin(); 
         it1 != oldPoints1.end(); it1++, it2++) {
        
//         if (((uint32_t)it1->y == (uint32_t)it2->y)) {
        if (fabs(it1->y - it2->y) < 1.0) {
            points1.push_back(*it1);
            points2.push_back(*it2);
        }
    }
}

inline 
void FundamentalMatrixEstimator::cleanMatchesBetweenFrames(vector< cv::Point2f >& points1, vector< cv::Point2f >& points2)
{
    const vector< cv::Point2f > oldPoints1 = points1;
    const vector< cv::Point2f > oldPoints2 = points2;
    
    points1.clear();
    points2.clear();
    
    points1.reserve(oldPoints1.size());
    points2.reserve(oldPoints2.size());
    
    for (vector<cv::Point2f>::const_iterator it1 = oldPoints1.begin(), it2 = oldPoints2.begin(); 
         it1 != oldPoints1.end(); it1++, it2++) {
        
        if (cv::norm(*it1 - *it2) < 10.0) {
            points1.push_back(*it1);
            points2.push_back(*it2);
        }
    }
}

inline 
void FundamentalMatrixEstimator::findPairCorrespondencesSURF(const cv::Mat& img1, const cv::Mat& img2, 
                                                             vector< cv::KeyPoint >& keypoints1, vector< cv::KeyPoint >& keypoints2,
                                                             const int & matchingMode)
{
    // SURF matching
    cv::SurfFeatureDetector surf(2500);
    cv::Mat desc1, desc2;
    
    bool firstImage = (keypoints1.size() == 0);
    
    if (firstImage) {
        cv::Mat mask1;
        cv::Canny(img1, mask1, 100, 200);
        cv::dilate(mask1, mask1, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
        
        surf(img1, mask1, keypoints1, desc1);
    }
    
    cv::Mat mask2;
    cv::Canny(img2, mask2, 100, 200);
    cv::dilate(mask2, mask2, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
    
    surf(img2, mask2, keypoints2, desc2);
    
    // Descriptors are matched
    vector<cv::DMatch> matches1, matches2;
    cv::FlannBasedMatcher matcher;
    matcher.match(desc1, desc2, matches1);
    
//     if (matchingMode == MATCH_STEREO_PAIR) {
//         cleanMatchesStereoPair(keypoints1, keypoints2, matches1, matches2);
//     } else if (matchingMode == MATCH_BETWEEN_FRAMES) {
//         cleanMatchesBetweenFrames(keypoints1, keypoints2, matches1, matches2);
//     }

//     cv::Mat imageMatches;
//     cv::drawMatches(img1, keypoints1, img2, keypoints2, matches2, imageMatches, cv::Scalar(255,255,255));
//     
//     cv::namedWindow("Matched");
//     cv::imshow("Matched", imageMatches);
//     
//     cv::waitKey();
}

inline 
void FundamentalMatrixEstimator::cleanMatchesStereoPair(const vector <cv::KeyPoint> & keypoints1, 
                                                        const vector <cv::KeyPoint> & keypoints2,
                                                        const vector<cv::DMatch> & matches1, vector<cv::DMatch> & matches2) {
    matches2.reserve(matches1.size());
    for (vector<cv::DMatch>::const_iterator it = matches1.begin(); it != matches1.end(); it++) {
        if (((uint32_t)keypoints2[it->trainIdx].pt.y == (uint32_t)keypoints1[it->queryIdx].pt.y))
            matches2.push_back(*it);
    }
}

inline 
void FundamentalMatrixEstimator::cleanMatchesBetweenFrames(const vector <cv::KeyPoint> & keypoints1, 
                                                            const vector <cv::KeyPoint> & keypoints2,
                                                            const vector<cv::DMatch> & matches1, vector<cv::DMatch> & matches2) {
    matches2.reserve(matches1.size());
    for (vector<cv::DMatch>::const_iterator it = matches1.begin(); it != matches1.end(); it++) {
        const double distance = cv::norm(keypoints2[it->trainIdx].pt - keypoints1[it->queryIdx].pt);
        if (distance < 10.0)
            matches2.push_back(*it);
    }
}

inline 
void FundamentalMatrixEstimator::cleanCorrespondences(const vector< vector< cv::Point2f > >& initialCorrespondences, 
                                                    vector< vector< cv::Point2f > >& finalCorrespondences)
{
    finalCorrespondences.resize(4);
    for (uint32_t i = 0; i < 4; i++) {
        finalCorrespondences[i].reserve(initialCorrespondences[0].size());
    }

    for (uint32_t i = 0; i < initialCorrespondences[0].size(); i++) {
        const cv::Point2f & p0 = initialCorrespondences[0][i];
        const cv::Point2f & p1 = initialCorrespondences[1][i];
        const cv::Point2f & p2 = initialCorrespondences[2][i];
        const cv::Point2f & p3 = initialCorrespondences[3][i];
        const cv::Point2f & p0b = initialCorrespondences[4][i];
    
        double dist = sqrt((p0.x - p0b.x) * (p0.x - p0b.x) + 
                           (p0.y - p0b.y) * (p0.y - p0b.y));
        if ((fabs(p0.y - p1.y) < MAX_HORIZONTAL_DIST) && (fabs(p2.y - p3.y) < MAX_HORIZONTAL_DIST) && 
//             (cv::norm(p1 - p2) < MAX_FLOW_DIST) && (cv::norm(p3 - p0b) < MAX_FLOW_DIST) &&
            (cv::norm(p0 - p0b) < MAX_CICLE_DIST)) {
            
            finalCorrespondences[0].push_back(p0);
            finalCorrespondences[1].push_back(p1);
            finalCorrespondences[2].push_back(p2);
            finalCorrespondences[3].push_back(p3);
        }
    
//     for (uint32_t i = 0; i < initialCorrespondences[0].size(); i++) {
//         const cv::Point2f & p1 = initialCorrespondences[0][i];
//         const cv::Point2f & p2 = initialCorrespondences[4][i];
//         
//         float dist = cv::norm(p1 - p2);
//         cout << p1 << " - " << p2 << ", dist " << dist << endl;
//         
//         if (dist < 1.0) {
//             finalCorrespondences[0].push_back(p1);
//             finalCorrespondences[1].push_back(initialCorrespondences[1][i]);
//             finalCorrespondences[2].push_back(initialCorrespondences[2][i]);
//             finalCorrespondences[3].push_back(initialCorrespondences[3][i]);
//         }
    }
}

inline 
void FundamentalMatrixEstimator::select8Points(vector< vector< cv::Point2f > >& correspondences, 
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
    for (uint32_t i = 0; i < finalCorrespondences[0].size(); i++)
        cout << "corresp " << finalCorrespondences[0][i] << endl;
}

inline
double FundamentalMatrixEstimator::measureFundMatrixQuality(const vector< cv::Point2f >& points1, const vector< cv::Point2f >& points2, const cv::Mat& F)
{
    vector<cv::Vec3f> lines1, lines2;
    
    cv::computeCorrespondEpilines(points1, 1, F, lines1);
    cv::computeCorrespondEpilines(points2, 2, F, lines2);
    
    double avgErr = 0;
    for(uint32_t i = 0; i < lines1.size(); i++ ) {
        const double err = fabs(points1[i].x*lines2[i][0] + points1[i].y*lines2[i][1] + lines2[i][2]) + 
        fabs(points2[i].x*lines1[i][0] + points2[i].y*lines1[i][1] + lines1[i][2]);
        avgErr += err;
    }
    
    return avgErr / (double)lines1.size();
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
    
//     for (uint32_t i = 0; i < correspondences[0].size(); i++) {
//         cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
//         
//         cv::circle(Rt0, correspondences[0][i], 5, color, -1);
//         cv::circle(Lt1, correspondences[1][i], 5, color, -1);
//         cv::circle(Rt1, correspondences[2][i], 5, color, -1);
//         cv::circle(Lt0, correspondences[3][i], 5, color, -1);
//     }
    
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
    
    for (uint32_t i = 0; i < finalCorrespondences[0].size(); i++) {
        cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        
//         cv::line(finalView, finalCorrespondences[0][i] * 0.625, (finalCorrespondences[1][i] + cv::Point2f(Lt0.cols, 0)) * 0.625, cv::Scalar(0, 0, 255));
//         cv::line(finalView, (finalCorrespondences[2][i] + cv::Point2f(Lt0.cols, Lt0.rows)) * 0.625, (finalCorrespondences[1][i] + cv::Point2f(Lt0.cols, 0)) * 0.625, cv::Scalar(0, 0, 255));
//         cv::line(finalView, (finalCorrespondences[2][i] + cv::Point2f(Lt0.cols, Lt0.rows)) * 0.625, (finalCorrespondences[3][i] + cv::Point2f(0, Lt0.rows)) * 0.625, cv::Scalar(0, 0, 255));
//         cv::line(finalView, finalCorrespondences[0][i] * 0.625, (finalCorrespondences[3][i] + cv::Point2f(0, Lt0.rows)) * 0.625, cv::Scalar(0, 0, 255));
        cv::circle(finalView, finalCorrespondences[0][i] * 0.625, 5, color, 1);
        cv::circle(finalView, (finalCorrespondences[1][i] + cv::Point2f(Lt0.cols, 0)) * 0.625, 5, color, 1);
        cv::circle(finalView, (finalCorrespondences[3][i] + cv::Point2f(0, Lt0.rows)) * 0.625, 5, color, 1);
        cv::circle(finalView, (finalCorrespondences[2][i] + cv::Point2f(Lt0.cols, Lt0.rows)) * 0.625, 5, color, 1);
        cv::line(finalView, finalCorrespondences[0][i] * 0.625, (finalCorrespondences[1][i] + cv::Point2f(Lt0.cols, 0)) * 0.625, color);
        cv::line(finalView, (finalCorrespondences[2][i] + cv::Point2f(Lt0.cols, Lt0.rows)) * 0.625, (finalCorrespondences[1][i] + cv::Point2f(Lt0.cols, 0)) * 0.625, color);
        cv::line(finalView, (finalCorrespondences[2][i] + cv::Point2f(Lt0.cols, Lt0.rows)) * 0.625, (finalCorrespondences[3][i] + cv::Point2f(0, Lt0.rows)) * 0.625, color);
        cv::line(finalView, finalCorrespondences[0][i] * 0.625, (finalCorrespondences[3][i] + cv::Point2f(0, Lt0.rows)) * 0.625, color);
        
    }
    
    cv::imshow("finalView", finalView);

    waitForKey();
}

void FundamentalMatrixEstimator::drawMatches(const cv::Mat& img1, const cv::Mat& img2, const vector< cv::Point2f >& points1, const vector< cv::Point2f >& points2)
{
    cv::Mat finalView(img1.rows, img1.cols * 2, CV_8UC3);
    
    img1.copyTo(finalView(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(finalView(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
    
    for (uint32_t i = 0; i < points1.size(); i++) {
        const cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        cv::circle(finalView, points1[i], 3, color);
        cv::circle(finalView, cv::Point2f(points2[i].x + img1.cols, points2[i].y), 3, color);
        
        cv::line(finalView, points1[i], cv::Point2f(points2[i].x + img1.cols, points2[i].y), cv::Scalar::all(255));
    }
    
    cv::imshow("matches", finalView);
}