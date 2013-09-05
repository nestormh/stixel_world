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


#ifndef STIXELSTRACKER_H
#define STIXELSTRACKER_H

#include "stereo_matching/stixels/motion/DummyStixelMotionEstimator.hpp"
#include "Eigen/Core"
#include <opencv2/opencv.hpp>
#include "polarcalibration.h"
#include <stixel_world_lib.hpp>

using namespace doppia;

namespace stixel_world {
    
class StixelsTracker : public DummyStixelMotionEstimator
{
public:    
    StixelsTracker(const boost::program_options::variables_map &options,
                    const MetricStereoCamera &camera, int stixels_width,
                   boost::shared_ptr<PolarCalibration> p_polarCalibration);
    void compute();
    
    void set_motion_cost_factors(const float & sad_factor, const float & height_factor, const float & polar_dist_factor);
    void drawTracker(cv::Mat & img);

protected:    
    static const uint8_t MAX_DISPARITY = 128;
    
    void compute_motion_cost_matrix();
    void transform_stixels_polar();
    cv::Point2d get_polar_point(const cv::Mat & mapX, const cv::Mat & mapY, const Stixel stixel);
    uint32_t compute_maximum_pixelwise_motion_for_stixel( const Stixel& stixel );
    void compute_maximum_pixelwise_motion_for_stixel_lut();
    void updateTracker();
    
    motion_cost_matrix_t m_stixelsPolarDistMatrix;
    Eigen::MatrixXi m_maximal_pixelwise_motion_by_disp;
    
    boost::shared_ptr<PolarCalibration> mp_polarCalibration;
    
    stixels_t m_previous_stixels_polar;
    stixels_t m_current_stixels_polar;
    
    float m_sad_factor; // SAD factor
    float m_height_factor; // height factor
    float m_polar_dist_factor; // polar dist factor
    
    typedef vector < stixels_t > t_tracker;
    t_tracker m_tracker;
};
}

#endif // STIXELSTRACKER_H
