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


#include "rectification.h"

#include<fstream>

using namespace stixel_world;

Rectification::Rectification()
{

}

Rectification::~Rectification()
{

}

void Rectification::readParamsFromFile(const string& fileName, const uint32_t& cameraId, const bool& readExtrinsic)
{
    ifstream fin(fileName.c_str(), ios::in);
    
    m_intrinsicCoeffs[cameraId] = cv::Mat(3, 3, CV_64FC1);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(0, 0);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(0, 1);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(0, 2);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(1, 0);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(1, 1);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(1, 2);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(2, 0);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(2, 1);
    fin >> m_intrinsicCoeffs[cameraId].at<double>(2, 2);
    
    m_distCoeffs[cameraId] = cv::Mat(1, 4, CV_64FC1);
    fin >> m_distCoeffs[cameraId].at<double>(0, 0);
    fin >> m_distCoeffs[cameraId].at<double>(0, 1);
    fin >> m_distCoeffs[cameraId].at<double>(0, 2);
    fin >> m_distCoeffs[cameraId].at<double>(0, 3);
    
    if (readExtrinsic) {
        m_R = cv::Mat(3, 3, CV_64FC1);
        fin >> m_R.at<double>(0, 0);
        fin >> m_R.at<double>(0, 1);
        fin >> m_R.at<double>(0, 2);
        fin >> m_R.at<double>(1, 0);
        fin >> m_R.at<double>(1, 1);
        fin >> m_R.at<double>(1, 2);
        fin >> m_R.at<double>(2, 0);
        fin >> m_R.at<double>(2, 1);
        fin >> m_R.at<double>(2, 2);
        
        m_t = cv::Mat(3, 1, CV_64FC1);
        fin >> m_t.at<double>(0, 0);
        fin >> m_t.at<double>(0, 1);
        fin >> m_t.at<double>(0, 2);
        fin >> m_t.at<double>(0, 3);
    }
    
    fin.close();
}

void Rectification::setIntrinsicCoeffs(const cv::Mat& intrinsicCoeffs, const uint32_t& cameraId)
{
    m_intrinsicCoeffs[cameraId] = intrinsicCoeffs;
}

void Rectification::setDistCoeffs(const cv::Mat& distCoeffs, const uint32_t& cameraId)
{
    m_distCoeffs[cameraId] = distCoeffs;
}

void Rectification::setRotationMatrix(const cv::Mat& R)
{
    m_R = R;
}

void Rectification::setTranslationMatrix(const cv::Mat& t)
{
    m_t = t;
}

bool Rectification::doRectification(const cv::Mat& img1, const cv::Mat& img2, 
                                    cv::Mat& rectified1, cv::Mat& rectified2, const uint32_t method)
{
    switch(method) {
        case RECTIFICATION_LINEAR:
            return doRectificationLinear(img1, img2, rectified1, rectified2);
//         case RECTIFICATION_POLAR:
            //TODO: Fundamental matrix could be found using a four-image scheme
//             return doRectificationPolar(img1, img2, rectified1, rectified2);
    }
    
    return false;
}

bool Rectification::doRectificationLinear(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& rectified1, cv::Mat& rectified2)
{
    assert((! m_intrinsicCoeffs[0].empty()) && (! m_intrinsicCoeffs[1].empty()) &&
            (! m_distCoeffs[0].empty()) && (! m_distCoeffs[1].empty()) &&
            (! m_R.empty()) && (! m_t.empty()));
    
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect roi1, roi2;
    cv::stereoRectify(m_intrinsicCoeffs[0], m_distCoeffs[0], m_intrinsicCoeffs[1], m_distCoeffs[1], img1.size(), 
                    m_R, m_t, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, -1, cv::Size(), &roi1, & roi2);
    
    cv::Mat mapX1, mapY1, mapX2, mapY2;
    cv::initUndistortRectifyMap(m_intrinsicCoeffs[0], m_distCoeffs[0], R1, P1, img1.size(), 
                                CV_32FC1, mapX1,  mapY1);
    cv::initUndistortRectifyMap(m_intrinsicCoeffs[1], m_distCoeffs[1], R2, P2, img1.size(), 
                                CV_32FC1, mapX2,  mapY2);
    cv::remap(img1, rectified1, mapX1, mapY1, cv::INTER_NEAREST);
    cv::remap(img2, rectified2, mapX2, mapY2, cv::INTER_NEAREST);
    
    rectified1 = rectified1(roi1);
    rectified2 = rectified2(roi2);
    
    return true;
}

uint32_t Rectification::getDisparityOffsetX()
{
    // 0, 3
    return m_intrinsicCoeffs[0].at<double>(0,3) - m_intrinsicCoeffs[1].at<double>(0,3);
}