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


#include "groundestimator.h"
#include "omp.h"

using namespace std;
using namespace stixel_world;

GroundEstimator::GroundEstimator(const Rectification & rectification) : m_rectification(rectification)
{
    m_justHalfImage = true;
    m_yStride = 1;
    m_maxDisparity = 128;
}

GroundEstimator::~GroundEstimator()
{

}

void GroundEstimator::setImagePair(const cv::Mat& img1, const cv::Mat& img2)
{
    if (m_justHalfImage) {
        m_left = img1(cv::Range(img1.rows / 2.0, img1.rows), cv::Range::all());
        m_right = img2(cv::Range(img2.rows / 2.0, img2.rows), cv::Range::all());
    } else {
        m_left = img1;
        m_right = img2;
    }

    m_disparity = cv::Mat::zeros(m_left.rows, m_maxDisparity, CV_16UC1);
    m_selectedPoints.clear();
    m_rowWeights.resize(m_disparity.rows);
}

bool GroundEstimator::compute()
{
    static int num_iterations = 0;
    static double cumulated_time = 0;
    
    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();
    
    // compute v_disparity --
    computeVDisparityData();
    
    vector<uint32_t> pointWeights;
    setPointsWeights(pointWeights);
    // compute line --
    estimateGroundPlane();
//     
//     confidence_is_up_to_date = false;
//     
//     // timing ---
//     cumulated_time += omp_get_wtime() - start_wall_time;
//     num_iterations += 1;
//     
//     if((silent_mode == false) and ((num_iterations % num_iterations_for_timing) == 0))
//     {
//         printf("Average FastGroundPlaneEstimator::compute speed  %.2lf [Hz] (in the last %i iterations)\n",
//                num_iterations / cumulated_time, 0/*num_iterations*/ );
//     }
    cout << "Time " << omp_get_wtime() - start_wall_time << endl;
    
    return true;
}

void GroundEstimator::computeVDisparityData()
{
    // for each pixel and each disparity value
    #pragma omp parallel for
    for(uint32_t rowIdx = 0; rowIdx < m_left.rows; rowIdx += m_yStride) {
        computeVDisparityRow(rowIdx);
    }
    
    //TODO: Debug
    cv::Mat visualizePoints = cv::Mat::zeros(m_disparity.rows, m_disparity.cols, CV_8UC3);
    for (uint32_t i = 0; i < m_selectedPoints.size(); i++)
        visualizePoints.at<cv::Vec3b>(m_selectedPoints[i].y, m_selectedPoints[i].x) = cv::Vec3b(0, 0, 255);
    
    cv::namedWindow("vdisparity");
    cv::imshow("vdisparity", m_disparity);
    cv::namedWindow("visualizePoints");
    cv::imshow("visualizePoints", visualizePoints);
    //TODO: End of Debug
    
    //printf("num_points == %i\n", points.size());
    return;
}

inline void GroundEstimator::computeVDisparityRow(const uint32_t& rowIdx)
{
    const int disparityOffset = m_rectification.getDisparityOffsetX();
    const uint16_t costSumSaturation = 5 * 3 * 16; // 5 * number_*of_pixels * levels_by_disparity
    assert(m_maxDisparity <= m_disparity.cols);
    
    uint16_t minCost = std::numeric_limits<uint16_t>::max();
    
    // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
    //const int first_right_x = first_left_x - disparity;
    for(uint32_t d = 0; d < m_maxDisparity; d ++) {
        uint16_t disparityCost = 0;
        
        for (uint32_t iL = d + disparityOffset, iR = 0; iL < m_left.cols; iL++, iR++) {
            const uint16_t cost = sad_cost_uint16(m_left.at<cv::Vec3b>(rowIdx, iL), m_right.at<cv::Vec3b>(rowIdx, iR));
            disparityCost += std::min(cost, costSumSaturation);
        }
        
        // we divide once at the end of the sums
        // this is ok to delay the division because
        // log2(1024*255*3) ~= 20 [bits]
        // so there is no risk of overflow inside 32bits
        disparityCost /= 3;
        m_disparity.at<uint16_t>(rowIdx, d) = disparityCost;
        
        minCost = std::min(disparityCost, minCost);
    } // end of "for each disparity"
    
    // select points to use for ground estimation --
    selectPointsAndWeights(rowIdx, minCost);
    
    return;
}

// compute the raw SAD (without doing a division by the number of channels)
inline uint16_t GroundEstimator::sad_cost_uint16(const cv::Vec3b& pixel_a, const cv::Vec3b& pixel_b)
{
    const int16_t delta_r = pixel_a[0] - pixel_b[0];
    const int16_t delta_g = pixel_a[1] - pixel_b[1];
    const int16_t delta_b = pixel_a[2] - pixel_b[2];
    
    // SAD
    const uint16_t distance = std::abs(delta_r) + std::abs(delta_g) + std::abs(delta_b);
    const uint16_t &cost = distance; // we skip the /3
    
    return cost;
}

inline void GroundEstimator::selectPointsAndWeights(const uint32_t& rowIdx, const uint16_t& minCost)
{
// vector<cv::Point2f> &points, vector<double> &row_weights
    const uint16_t & threshCost = minCost + DELTA_COST;
    uint32_t pointsInRow = 0;
    
    for(uint32_t d = 0; d < m_maxDisparity; d++) {
        if (m_disparity.at<uint16_t>(rowIdx, d) <= threshCost) {
            #pragma omp critical
            {
                m_selectedPoints.push_back( cv::Point2f(d, rowIdx) );
            }
            pointsInRow++;
            
            if(pointsInRow > MAX_POINTS_IN_ROW) {
                // this line is useless, no need to collect more points
                break;
            }
        } else {
           // we discard this point
            continue;
        }
    } // end of "for each disparity"
    
    if(pointsInRow > 0) {
        // rows with less points give more confidence
        m_rowWeights[rowIdx] = 1.0 / (double)pointsInRow;
    }
    
    return;
}

void GroundEstimator::setPointsWeights(vector<uint32_t> & pointWeights)
{
    pointWeights.resize(m_selectedPoints.size());
    
    for (uint32_t i = 0; i < m_selectedPoints.size(); i++) {
        pointWeights[i] = m_rowWeights[m_selectedPoints[i].y];
    }
        
    return;
}

void GroundEstimator::estimateGroundPlane()
{
    // TODO
}
