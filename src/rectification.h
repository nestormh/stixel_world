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


#ifndef RECTIFICATION_H
#define RECTIFICATION_H

#include<string.h>
#include<opencv2/opencv.hpp>

using namespace std;

namespace stixel_world {
class Rectification
{

public:
    static const uint32_t RECTIFICATION_LINEAR = 0;
    static const uint32_t RECTIFICATION_POLAR = 1;
    
    Rectification();
    virtual ~Rectification();
    void readParamsFromFile(const string & fileName, const uint32_t & cameraId, const bool & readExtrinsic = false);
    
    void setIntrinsicCoeffs(const cv::Mat & intrinsicCoeffs, const uint32_t & cameraId);
    void setDistCoeffs(const cv::Mat & distCoeffs, const uint32_t & cameraId);
    void setRotationMatrix(const cv::Mat & R);
    void setTranslationMatrix(const cv::Mat & t);
    
    bool doRectification(const cv::Mat & img1, const cv::Mat & img2, 
                         cv::Mat & rectified1, cv::Mat & rectified2, const uint32_t method = RECTIFICATION_LINEAR);
    
    cv::Mat getIntrinsicCoeffs(const uint32_t & cameraId) { return m_intrinsicCoeffs[cameraId]; }
    cv::Mat getDistCoeffs(const uint32_t & cameraId) { return m_distCoeffs[cameraId]; }
    cv::Mat getR() { return m_R; }
    cv::Mat getT() { return m_t; }
    uint32_t getDisparityOffsetX();
    
private:
    bool doRectificationLinear(const cv::Mat & img1, const cv::Mat & img2, 
                               cv::Mat & rectified1, cv::Mat & rectified2);
    
    cv::Mat m_intrinsicCoeffs[2];
    cv::Mat m_distCoeffs[2];
    cv::Mat m_R;
    cv::Mat m_t;
};
}
#endif // RECTIFICATION_H
