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
#include "doppia/stixel3d.h"
#include "densetracker.h"

using namespace doppia;

namespace stixel_world {
    
class StixelsTracker : public DummyStixelMotionEstimator
{
public:    
    StixelsTracker(const boost::program_options::variables_map &options,
                    const MetricStereoCamera &camera, int stixels_width,
                   boost::shared_ptr<PolarCalibration> p_polarCalibration);
    void compute();
    
    void set_motion_cost_factors(const float & sad_factor, const float & height_factor, 
                                 const float & polar_dist_factor, const float & polar_sad_factor,
                                 const float & dense_tracking_factor, const float & hist_similarity_factor, 
                                 const bool & useGraphs);
    
    void updateDenseTracker(const cv::Mat & frame);
    
    void drawTracker(cv::Mat & img, cv::Mat & imgTop);
    void drawTracker(cv::Mat & img);
    void drawDenseTracker(cv::Mat & img);
    
    float getSADFactor() { return m_sad_factor; }
    float getHeightFactor() { return m_height_factor; }
    float getPolarDistFactor() { return m_polar_dist_factor; }
    float getPolarSADFactor() { return m_polar_sad_factor; }
    float getDenseTrackingFactor() { return m_dense_tracking_factor; }
    bool useGraphs() { return m_useGraphs; }
    
    typedef vector < stixels3d_t > t_tracker;
    typedef deque <stixels3d_t> t_historic;
    t_tracker getTracker() { return m_tracker; }
    t_historic getHistoric() { return m_stixelsHistoric; }
    
    stixels3d_t getLastStixelsAfterTracking();

protected:    
    static const uint8_t MAX_DISPARITY = 128;
    static const uint8_t MAX_ITERATIONS_STORED = 51;
    
    typedef struct {
        cv::Rect roi;
        double z;
        stixels3d_t stixels;
    } t_obstacle;
    
    void estimate_stixel_direction();
    void compute_static_stixels();
    void compute_motion_cost_matrix();
    void transform_stixels_polar();
    cv::Point2d get_polar_point(const cv::Mat & mapX, const cv::Mat & mapY, const Stixel & stixel, const bool bottom = true);
    cv::Point2d get_polar_point(const cv::Mat & prevMapX, const cv::Mat & prevMapY, 
                                const cv::Mat & currPolar2LinearX, const cv::Mat & currPolar2LinearY, 
                                const Stixel & stixel);
    cv::Point2d get_polar_point(const cv::Mat& mapX, const cv::Mat& mapY, const cv::Point2d & point);
    uint32_t compute_maximum_pixelwise_motion_for_stixel( const Stixel& stixel );
    void compute_maximum_pixelwise_motion_for_stixel_lut();
    void updateTracker();
    void getClusters();
    float compute_polar_SAD(const Stixel& stixel1, const Stixel& stixel2);
    float compute_polar_SAD(const Stixel& stixel1, const Stixel& stixel2,
                            const input_image_const_view_t& image_view1, const input_image_const_view_t& image_view2,
                            const unsigned int stixel_horizontal_padding);
    void compute_stixel_representation_polar( const Stixel &stixel, const input_image_const_view_t& image_view_hosting_the_stixel,
                                               stixel_representation_t &stixel_representation, const unsigned int stixel_horizontal_padding,
                                              const cv::Mat & mapX, const cv::Mat & mapY, const cv::Mat & polarImg);
    float compute_dense_tracking_score(const Stixel& currStixel, const Stixel& prevStixel);
    void draw_polar_SAD(cv::Mat & img, const Stixel& stixel1, const Stixel& stixel2);
    
    void projectPointInTopView(const cv::Point3d & point3d, const cv::Mat & imgTop, cv::Point2d & point2d);
    
    void computeMotionWithGraphs();
    
    void computeHistogram(cv::Mat & hist, const cv::Mat & img, const Stixel & stixel);
    float compareHistogram(const cv::Mat& hist1, const cv::Mat& hist2, const Stixel& stixel1, const Stixel& stixel2);
    void correctStixelsUsingTracking();
    
    motion_cost_matrix_t m_stixelsPolarDistMatrix;
    motion_cost_matrix_t m_polarSADMatrix;
    motion_cost_matrix_t m_denseTrackingMatrix;
    motion_cost_matrix_t m_histogramComparisonMatrix;
    Eigen::MatrixXi m_maximal_pixelwise_motion_by_disp;
    
    boost::shared_ptr<PolarCalibration> mp_polarCalibration;
    boost::shared_ptr<dense_tracker::DenseTracker> mp_denseTracker;
    
    stixels_t m_previous_stixels_polar;
    stixels_t m_current_stixels_polar;
    
    cv::Mat m_mapXprev, m_mapYprev, m_mapXcurr, m_mapYcurr;
    cv::Mat m_polarImg1, m_polarImg2;
    
    float m_sad_factor; // SAD factor
    float m_height_factor; // height factor
    float m_polar_dist_factor; // polar dist factor
    float m_polar_sad_factor;  // SAD in polar images
    float m_dense_tracking_factor; // Dense tracking
    float m_hist_similarity_factor; // Histogram similarity
    
    bool m_useGraphs;
    
    float m_minPolarSADForBeingStatic;
    
    t_tracker m_tracker;
    t_historic m_stixelsHistoric;
    
    vector<cv::Scalar> m_color;
    
    vector<int32_t> m_clusters;
    vector < vector<int> > m_objects;
    
    vector < t_obstacle> m_obstacles;
    
    double m_minAllowedObjectWidth;
    double m_minDistBetweenClusters;
    
    cv::Mat m_currImg;
};
}

#endif // STIXELSTRACKER_H
