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

// NOTE: A big part of this class is an adaptation to opencv of the code available at:
// https://bitbucket.org/rodrigob/doppia

#ifndef GROUNDESTIMATOR_H
#define GROUNDESTIMATOR_H
#include <opencv2/opencv.hpp>
#include "rectification.h"

#define MAX_POINTS_IN_ROW 20
#define DELTA_COST 1 //2 //5

namespace stixel_world {
class GroundEstimator
{

public:
    GroundEstimator(const Rectification & rectification);
    virtual ~GroundEstimator();
    void setImagePair(const cv::Mat & img1, const cv::Mat & img2);
    bool compute();
    
    void toggleJustHalfImage(const bool & justHalfImage) { m_justHalfImage = justHalfImage; }
    void setYStride(const double & yStride) { m_yStride = yStride; }
    void setMaxDisparity(const uint32_t & maxDisparity) { m_maxDisparity = maxDisparity; }
private:
    
    void computeVDisparityData();
    void computeVDisparityRow(const uint32_t & rowIdx);
    uint16_t sad_cost_uint16(const cv::Vec3b &pixel_a, const cv::Vec3b  &pixel_b);
    void selectPointsAndWeights(const uint32_t & rowIdx, const uint16_t & minCost);
    void setPointsWeights(vector<uint32_t> & pointWeights);
    void estimateGroundPlane();
    
    cv::Mat m_left, m_right, m_disparity;
    bool m_justHalfImage;
    uint32_t m_yStride;
    uint32_t m_maxDisparity;
    Rectification m_rectification;
    vector<cv::Point2f> m_selectedPoints;
    vector<double> m_rowWeights;
};
}

#endif // GROUNDESTIMATOR_H
