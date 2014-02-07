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


#include "stixelstracker.h"

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <boost/graph/graph_concepts.hpp>

#include<boost/foreach.hpp>
#include <tiff.h>
#include <lemon/matching.h>
#include <lemon/smart_graph.h>

#include <algorithm>

#include <ros/ros.h>

#include "kalmanfilter.h"

#include "utils.h"

using namespace std;
using namespace stixel_world;

const float MIN_FLOAT_DISPARITY = 0.8f;


StixelsTracker::StixelsTracker::StixelsTracker(const boost::program_options::variables_map& options, 
                                               const MetricStereoCamera& camera, int stixels_width,
                                               boost::shared_ptr<PolarCalibration> p_polarCalibration) :
                                               DummyStixelMotionEstimator(options, camera, stixels_width),
                                               mp_polarCalibration(p_polarCalibration)
{ 
    m_stixelsPolarDistMatrix = Eigen::MatrixXf::Zero( motion_cost_matrix.rows(), motion_cost_matrix.cols() ); // Matrix is initialized with 0.
    m_polarSADMatrix = Eigen::MatrixXf::Zero( motion_cost_matrix.rows(), motion_cost_matrix.cols() ); // Matrix is initialized with 0.
    m_denseTrackingMatrix = Eigen::MatrixXf::Zero( motion_cost_matrix.rows(), motion_cost_matrix.cols() ); // Matrix is initialized with 0.
    m_histogramComparisonMatrix = Eigen::MatrixXf::Zero( motion_cost_matrix.rows(), motion_cost_matrix.cols() ); // Matrix is initialized with 0.
    compute_maximum_pixelwise_motion_for_stixel_lut();
    
    m_sad_factor = 0.3f;
    m_height_factor = 0.0f;
    m_polar_dist_factor = 0.0f;
    m_polar_sad_factor = 0.0f;
    m_dense_tracking_factor = 0.7f;
    m_hist_similarity_factor = 0.0f;
    
    m_minAllowedObjectWidth = 0.3;
    m_minDistBetweenClusters = 0.3;
    
    m_minPolarSADForBeingStatic = 10;
    
    m_useGraphs = true;
    
//     mp_denseTracker.reset(new dense_tracker::DenseTracker());
}

void StixelsTracker::set_motion_cost_factors(const float& sad_factor, const float& height_factor, 
                                             const float& polar_dist_factor, const float & polar_sad_factor,
                                             const float& dense_tracking_factor, const float & hist_similarity_factor, 
                                             const bool & useGraphs, const bool & useCostMatrix, const bool & useObjects)
{
    if ((sad_factor + height_factor + polar_dist_factor + polar_sad_factor + dense_tracking_factor + hist_similarity_factor) == 1.0) {
        m_sad_factor = sad_factor;
        m_height_factor = height_factor;
        m_polar_dist_factor = polar_dist_factor;
        m_polar_sad_factor = polar_sad_factor;
        m_dense_tracking_factor = dense_tracking_factor;
        m_hist_similarity_factor = hist_similarity_factor;
        m_useCostMatrix = useCostMatrix;
        m_useObjects = useObjects;
        if (m_dense_tracking_factor != 0)
            mp_denseTracker.reset(new dense_tracker::DenseTracker());
    } else {
        cerr << "The sum of motion cost factors should be 1!!!" << endl;
        exit(0);
    }
    
    m_useGraphs = useGraphs;
}

void StixelsTracker::transform_stixels_polar()
{
    cv::Mat mapXprev, mapYprev, mapXcurr, mapYcurr;
    mp_polarCalibration->getInverseMaps(mapXprev, mapYprev, 1);
    mp_polarCalibration->getInverseMaps(mapXcurr, mapYcurr, 2);
    
    m_previous_stixels_polar.clear();
    m_current_stixels_polar.clear();
    
    m_previous_stixels_polar.resize(previous_stixels_p->size());
    m_current_stixels_polar.resize(current_stixels_p->size());
    
    copy(previous_stixels_p->begin(), previous_stixels_p->end(), m_previous_stixels_polar.begin());
    copy(current_stixels_p->begin(), current_stixels_p->end(), m_current_stixels_polar.begin());
    
    for (stixels_t::iterator it = m_previous_stixels_polar.begin(); it != m_previous_stixels_polar.end(); it++) {
        const cv::Point2d newPos(mapXprev.at<float>(it->bottom_y, it->x),
                                 mapYprev.at<float>(it->bottom_y, it->x));
        it->x = newPos.x;
        it->bottom_y = newPos.y;
    }
    
    for (stixels_t::iterator it = m_current_stixels_polar.begin(); it != m_current_stixels_polar.end(); it++) {
        const cv::Point2d newPos(mapXcurr.at<float>(it->bottom_y, it->x),
                                 mapYcurr.at<float>(it->bottom_y, it->x));
        it->x = newPos.x;
        it->bottom_y = newPos.y;
    }
}

inline
cv::Point2d StixelsTracker::get_polar_point(const cv::Mat& mapX, const cv::Mat& mapY, const Stixel & stixel, const bool bottom)
{
    if (bottom)
        return cv::Point2d(mapX.at<float>(stixel.bottom_y, stixel.x),
                       mapY.at<float>(stixel.bottom_y, stixel.x));
    else
        return cv::Point2d(mapX.at<float>(stixel.top_y, stixel.x),
                           mapY.at<float>(stixel.top_y, stixel.x));
}

inline
cv::Point2d StixelsTracker::get_polar_point(const cv::Mat& prevMapX, const cv::Mat& prevMapY, 
                                            const cv::Mat& currPolar2LinearX, const cv::Mat& currPolar2LinearY, const Stixel& stixel)
{
    const cv::Point2d polarPoint(prevMapX.at<float>(stixel.bottom_y, stixel.x),
                                 prevMapY.at<float>(stixel.bottom_y, stixel.x));
    
    if (polarPoint == cv::Point2d(-1, -1))
        return polarPoint;
    
    return cv::Point2d(currPolar2LinearX.at<float>(polarPoint.y, polarPoint.x),
                       currPolar2LinearY.at<float>(polarPoint.y, polarPoint.x));
}


inline
cv::Point2d StixelsTracker::get_polar_point(const cv::Mat& mapX, const cv::Mat& mapY, const cv::Point2d & point)
{
    return cv::Point2d(mapX.at<float>(point.y, point.x),
                       mapY.at<float>(point.y, point.x));
}

void StixelsTracker::updateDenseTracker(const cv::Mat & frame)
{
    if (m_dense_tracking_factor != 0.0f)
        mp_denseTracker->compute(frame);
}

void StixelsTracker::compute()
{
//     DummyStixelMotionEstimator::compute();
//     updateTracker();
    
//     compute_motion_cost_matrix();
//     compute_motion_v1();
//     updateTracker();
    
//     compute_static_stixels();
//     compute_motion_cost_matrix();
//     if (m_useGraphs) {
//         computeMotionWithGraphs();
// //         computeMotionWithGraphsAndHistogram();
//     } else {
//         compute_motion();
//     }
// //     update_stixel_tracks_image();
//     trackObstacles();
//     
//     updateTracker();
// //     estimate_stixel_direction();
// //     getClusters();
    ///////////////////////////////////////
double startWallTime = omp_get_wtime();
    if (m_useCostMatrix) {
        compute_motion_cost_matrix();
    }
    if (m_useGraphs) {
        computeMotionWithGraphs();
    } else {
        compute_motion();
    }
    if (m_useObjects) {
        trackObstacles();
    } else {
        update_stixel_tracks_image();
        updateTracker();
    }

// #if 0    // New tracking style
//     
//     compute_motion_cost_matrix();
//     if (m_useGraphs) {
//         computeMotionWithGraphs();
// //         computeMotionWithGraphsAndHistogram();
//     } else {
//         compute_motion();
//     }
//     //     update_stixel_tracks_image();
//     trackObstacles();
//     
// //     updateTracker();
// #else // Old tracking style
//         compute_motion_cost_matrix();
// //     if (m_useGraphs) {
// //         computeMotionWithGraphs();
// //         computeMotionWithGraphsAndHistogram();
// //     } else {
//         compute_motion();
// //     }
//     update_stixel_tracks_image();
// //     trackObstacles();
// 
//     updateTracker();
// #endif
ROS_ERROR("[TIMES] Time %f", omp_get_wtime() - startWallTime);
    
    return;
}

void StixelsTracker::compute_motion_cost_matrix()
{    
    
    const double & startWallTime = omp_get_wtime();
    
    const float maximum_depth_difference = 1.0;
    
    const float maximum_allowed_real_height_difference = 0.5f;
    const float maximum_allowed_polar_distance = 50.0f;
    assert((m_sad_factor + m_height_factor + m_polar_dist_factor + m_polar_sad_factor + m_dense_tracking_factor + m_hist_similarity_factor) == 1.0f);
    
    const float maximum_real_motion = maximum_pedestrian_speed / video_frame_rate;
    
    const unsigned int number_of_current_stixels = current_stixels_p->size();
    const unsigned int number_of_previous_stixels = previous_stixels_p->size();

    cv::Mat currPolar2LinearX, currPolar2LinearY;
    if (mp_polarCalibration) {
        mp_polarCalibration->getStoredRectifiedImages(m_polarImg1, m_polarImg2);
        
        mp_polarCalibration->getInverseMaps(m_mapXprev, m_mapYprev, 1);
        mp_polarCalibration->getInverseMaps(m_mapXcurr, m_mapYcurr, 2);
        
        mp_polarCalibration->getMaps(currPolar2LinearX, currPolar2LinearY, 2);
    }
    
    motion_cost_matrix.fill( 0.f );
    pixelwise_sad_matrix.fill( 0.f );
    real_height_differences_matrix.fill( 0.f );
    m_stixelsPolarDistMatrix.fill(0.f);
    m_polarSADMatrix.fill(0.f);
    m_denseTrackingMatrix.fill(0.f);
    m_histogramComparisonMatrix.fill(0.f);
    motion_cost_assignment_matrix.fill( false );
    
    current_stixel_depths.fill( 0.f );
    current_stixel_real_heights.fill( 0.f );
    
    cv::MatND hist1, hist2;
    cv::Mat lastImg = m_currImg;
    if (lastImg.empty())
        gil2opencv(previous_image_view, lastImg);
    gil2opencv(current_image_view, m_currImg);
    
    // Fill in the motion cost matrix
//     #pragma omp parallel for schedule(dynamic)
    for( unsigned int s_current = 0; s_current < number_of_current_stixels; ++s_current )
    {
        const Stixel& current_stixel = ( *current_stixels_p )[ s_current ];
//         const cv::Point2d current_polar = get_polar_point(mapXcurr, mapYcurr, current_stixel);
        cv::Point2d current_polar;
        if (mp_polarCalibration)
            current_polar = get_polar_point(m_mapXcurr, m_mapYcurr, currPolar2LinearX, currPolar2LinearY, current_stixel);
        
        if (m_hist_similarity_factor != 0.0) computeHistogram(hist1, m_currImg, current_stixel);
        
        const unsigned int stixel_horizontal_padding = compute_stixel_horizontal_padding( current_stixel );
        
        /// Do NOT add else conditions since it can affect the computation of matrices
        if( current_stixel.x - ( current_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 &&
            current_stixel.x + ( current_stixel.width - 1 ) / 2 + stixel_horizontal_padding < current_image_view.width() /*&&
            current_stixel.type != Stixel::Occluded*/ ) // Horizontal padding for current stixel is suitable
        {
            const float current_stixel_disparity = std::max< float >( MIN_FLOAT_DISPARITY, current_stixel.disparity );
            const float current_stixel_depth = stereo_camera.disparity_to_depth( current_stixel_disparity );
            
            const float current_stixel_real_height = compute_stixel_real_height( current_stixel );
            
            // Store for future reference
            current_stixel_depths( s_current ) = current_stixel_depth;
            current_stixel_real_heights( s_current ) = current_stixel_real_height;
            
            for( unsigned int s_prev = 0; s_prev < number_of_previous_stixels; ++s_prev )
            {
                const Stixel& previous_stixel = ( *previous_stixels_p )[ s_prev ];
//                 const cv::Point2d previous_polar = get_polar_point(mapXprev, mapYprev, previous_stixel);
                cv::Point2d previous_polar;
                if (mp_polarCalibration)
                    previous_polar = get_polar_point(m_mapXprev, m_mapYprev, currPolar2LinearX, currPolar2LinearY, previous_stixel);
                
                if (m_hist_similarity_factor != 0.0) computeHistogram(hist2, lastImg, previous_stixel);
                
                if( previous_stixel.x - ( previous_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 &&
                    previous_stixel.x + ( previous_stixel.width - 1 ) / 2 + stixel_horizontal_padding < previous_image_view.width())
                {
                    const float previous_stixel_disparity = std::max< float >( MIN_FLOAT_DISPARITY, previous_stixel.disparity );
                    const float previous_stixel_depth = stereo_camera.disparity_to_depth( previous_stixel_disparity );
                    
                    if( fabs( current_stixel_depth - previous_stixel_depth ) < maximum_depth_difference )
                    {
                        const int pixelwise_motion = previous_stixel.x - current_stixel.x; // Motion can be positive or negative
                        
                        const unsigned int maximum_motion_in_pixels_for_current_stixel = compute_maximum_pixelwise_motion_for_stixel( current_stixel );
                        
                        if( pixelwise_motion >= -( int( maximum_motion_in_pixels_for_current_stixel ) ) &&
                            pixelwise_motion <= int( maximum_motion_in_pixels_for_current_stixel ))
                        {
                            float pixelwise_sad;
                            float real_height_difference;
                            float polar_distance;
                            float polar_SAD;
                            float denseTrackingScore;
                            float histogramComparisonScore;
                            
                            if( current_stixel.type != Stixel::Occluded && previous_stixel.type != Stixel::Occluded )
                            {
                                pixelwise_sad = (m_sad_factor == 0.0f)? 0.0f : compute_pixelwise_sad( current_stixel, previous_stixel, current_image_view, previous_image_view, stixel_horizontal_padding );
                                real_height_difference = (m_height_factor == 0.0f)? 0.0f : fabs( current_stixel_real_height - compute_stixel_real_height( previous_stixel ) );
                                polar_distance = (m_polar_dist_factor == 0.0f)? 0.0f : cv::norm(previous_polar - current_polar);
//                                 polar_SAD = (m_polar_sad_factor == 0.0f)? 0.0f : compute_polar_SAD(current_stixel, previous_stixel, current_image_view, previous_image_view, stixel_horizontal_padding);
                                polar_SAD = (m_polar_sad_factor == 0.0f)? 0.0f : compute_polar_SAD(current_stixel, previous_stixel);
                                denseTrackingScore = (m_dense_tracking_factor == 0.0f)? 0.0f : compute_dense_tracking_score(current_stixel, previous_stixel);
                                histogramComparisonScore = (m_hist_similarity_factor == 0.0)? 0.0f : compareHistogram(hist1, hist2, current_stixel, previous_stixel);
                            }
                            else
                            {
                                pixelwise_sad = maximum_pixel_value;
                                real_height_difference = maximum_allowed_real_height_difference;
                                polar_distance = maximum_allowed_polar_distance;
                                polar_SAD = maximum_pixel_value;
                                denseTrackingScore = maximum_pixel_value;
                                histogramComparisonScore = maximum_pixel_value;
                            }
                            
                            pixelwise_sad_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = pixelwise_sad;
                            real_height_differences_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) =
                                    std::min( 1.0f, real_height_difference / maximum_allowed_real_height_difference );
                            
                            m_stixelsPolarDistMatrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = 1.0f -
                                    std::min( 1.0f, polar_distance / maximum_allowed_polar_distance );
                                    
                            m_polarSADMatrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = maximum_pixel_value - polar_SAD;
                            
                            m_denseTrackingMatrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = denseTrackingScore;
                            m_histogramComparisonMatrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = histogramComparisonScore * 255.0;
                            
                            motion_cost_assignment_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = true;
                            
//                             if (polar_distance > 5.0)
//                                 motion_cost_assignment_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = false;
                        }
                    }
                }
                
            } // End of for( s_prev )
        }
        
    } // End of for( s_current )
   
    /// Rescale the real height difference matrix elemants so that it will have the same range with pixelwise_sad
    const float maximum_real_height_difference = real_height_differences_matrix.maxCoeff();
    //    real_height_differences_matrix = real_height_differences_matrix * ( float ( maximum_pixel_value ) / maximum_real_height_difference );
//     real_height_differences_matrix = real_height_differences_matrix * maximum_pixel_value;
    real_height_differences_matrix = real_height_differences_matrix * (maximum_pixel_value / maximum_real_height_difference);
    
    const float maximum_dense_tracking_value = m_denseTrackingMatrix.maxCoeff();
    m_denseTrackingMatrix = m_denseTrackingMatrix * (maximum_pixel_value / maximum_dense_tracking_value);
    for (uint32_t i = 0; i < m_denseTrackingMatrix.rows(); i++) {
        for (uint32_t j = 0; j < m_denseTrackingMatrix.cols(); j++) {
            m_denseTrackingMatrix(i, j) = maximum_pixel_value - m_denseTrackingMatrix(i, j);
        }
    }
    
    const float maximum_polar_dist_value = m_stixelsPolarDistMatrix.maxCoeff();
    m_stixelsPolarDistMatrix = m_stixelsPolarDistMatrix * (maximum_pixel_value / maximum_polar_dist_value);
    
    /// Fill in the motion cost matrix
//     motion_cost_matrix = alpha * pixelwise_sad_matrix + ( 1 - alpha ) * real_height_differences_matrix; // [0, 255]
    motion_cost_matrix = m_sad_factor * pixelwise_sad_matrix + 
                         m_height_factor * real_height_differences_matrix +
                         m_polar_dist_factor * m_stixelsPolarDistMatrix + 
                         m_polar_sad_factor * m_polarSADMatrix +
                         m_dense_tracking_factor * m_denseTrackingMatrix +
                         m_hist_similarity_factor * m_histogramComparisonMatrix;
                         
    /*{
        cv::Mat scores(m_denseTrackingMatrix.rows(), m_denseTrackingMatrix.cols(), CV_8UC1);
        for (uint32_t i = 0; i < m_denseTrackingMatrix.rows(); i++) {
            for (uint32_t j = 0; j < m_denseTrackingMatrix.cols(); j++) {
                scores.at<uchar>(i, j) = m_denseTrackingMatrix(i, j);
            }
        }
        cv::imshow("scoresDT", scores);
    }
    {
        cv::Mat scores(pixelwise_sad_matrix.rows(), pixelwise_sad_matrix.cols(), CV_8UC1);
        for (uint32_t i = 0; i < pixelwise_sad_matrix.rows(); i++) {
            for (uint32_t j = 0; j < pixelwise_sad_matrix.cols(); j++) {
                scores.at<uchar>(i, j) = pixelwise_sad_matrix(i, j);
            }
        }
        cv::imshow("pixelwise_sad_matrix", scores);
    }
    {
        cv::Mat scores(real_height_differences_matrix.rows(), real_height_differences_matrix.cols(), CV_8UC1);
        for (uint32_t i = 0; i < real_height_differences_matrix.rows(); i++) {
            for (uint32_t j = 0; j < real_height_differences_matrix.cols(); j++) {
                scores.at<uchar>(i, j) = real_height_differences_matrix(i, j);
            }
        }
        cv::imshow("real_height_differences_matrix", scores);
    }
    {
        cv::Mat scores(m_stixelsPolarDistMatrix.rows(), m_stixelsPolarDistMatrix.cols(), CV_8UC1);
        for (uint32_t i = 0; i < m_stixelsPolarDistMatrix.rows(); i++) {
            for (uint32_t j = 0; j < m_stixelsPolarDistMatrix.cols(); j++) {
                scores.at<uchar>(i, j) = m_stixelsPolarDistMatrix(i, j);
            }
        }
        cv::imshow("m_stixelsPolarDistMatrix", scores);
    }
    {
        cv::Mat scores(m_polarSADMatrix.rows(), m_polarSADMatrix.cols(), CV_8UC1);
        for (uint32_t i = 0; i < m_polarSADMatrix.rows(); i++) {
            for (uint32_t j = 0; j < m_polarSADMatrix.cols(); j++) {
                scores.at<uchar>(i, j) = m_polarSADMatrix(i, j);
            }
        }
        cv::imshow("m_polarSADMatrix", scores);
    }
    {
        cv::Mat scores(motion_cost_matrix.rows(), motion_cost_matrix.cols(), CV_8UC1);
        for (uint32_t i = 0; i < motion_cost_matrix.rows(); i++) {
            for (uint32_t j = 0; j < motion_cost_matrix.cols(); j++) {
                scores.at<uchar>(i, j) = motion_cost_matrix(i, j);
            }
        }
        cv::imshow("scoresMotionCost", scores);
    }*/
                         
    const float maximum_cost_matrix_element = motion_cost_matrix.maxCoeff(); // Minimum is 0 by definition
    
    /// Fill in disappearing stixel entries specially
    //    insertion_cost_dp = maximum_cost_matrix_element * 0.75;
    insertion_cost_dp = maximum_pixel_value * 0.6;
    deletion_cost_dp = insertion_cost_dp; // insertion_cost_dp is not used for the moment !!
    
    {
//         const unsigned int number_of_cols = motion_cost_matrix.cols();
//         const unsigned int largest_row_index = motion_cost_matrix.rows() - 1;
        
    for( unsigned int j = 0, number_of_cols = motion_cost_matrix.cols(), largest_row_index = motion_cost_matrix.rows() - 1; j < number_of_cols; ++j )
//         #pragma omp parallel for schedule(dynamic)
        for( unsigned int j = 0; j < number_of_cols; ++j )
        {
            motion_cost_matrix( largest_row_index, j ) = deletion_cost_dp;
            motion_cost_assignment_matrix( largest_row_index, j ) = true;
            
        } // End of for(j)
    }
    
    {
//         const unsigned int number_of_rows = motion_cost_matrix.rows();
//         const unsigned int number_of_cols = motion_cost_matrix.cols();
                
        for( unsigned int i = 0, number_of_rows = motion_cost_matrix.rows(); i < number_of_rows; ++i )
//         #pragma omp parallel for schedule(dynamic)
//         for( unsigned int i = 0; i < number_of_rows; ++i )
        {
            for( unsigned int j = 0, number_of_cols = motion_cost_matrix.cols(); j < number_of_cols; ++j )
            {
                if( motion_cost_assignment_matrix( i, j ) == false )
                {
                    motion_cost_matrix( i, j ) = 1.2 * maximum_cost_matrix_element;
                    // motion_cost_assignment_matrix(i,j) should NOT be set to true for the entries which are "forced".
                }
            }
        }    
    }
    
    /**
     * 
     * Lines below are intended for DEBUG & VISUALIZATION purposes
     *
     **/
    
    //    fill_in_visualization_motion_cost_matrix();
    cout << "Time for " << __FUNCTION__ << ":" << __LINE__ << " " << omp_get_wtime() - startWallTime << endl;
    
    return;
}

void StixelsTracker::compute_maximum_pixelwise_motion_for_stixel_lut( ) 
{
    m_maximal_pixelwise_motion_by_disp = Eigen::MatrixXi::Zero(MAX_DISPARITY, 1);
    for (uint32_t disp = 0; disp < MAX_DISPARITY; disp++) {
        float disparity = std::max< float >( MIN_FLOAT_DISPARITY, disp );
        float depth = stereo_camera.disparity_to_depth( disparity );

        Eigen::Vector3f point3d1( -maximum_displacement_between_frames / 2, 0, depth );
        Eigen::Vector3f point3d2( maximum_displacement_between_frames / 2, 0, depth );
        
        const MetricCamera& left_camera = stereo_camera.get_left_camera();
        
        Eigen::Vector2f point2d1 = left_camera.project_3d_point( point3d1 );
        Eigen::Vector2f point2d2 = left_camera.project_3d_point( point3d2 );
        
        m_maximal_pixelwise_motion_by_disp(disp, 0) = static_cast<unsigned int>( fabs( point2d2[ 0 ] - point2d1[ 0 ] ) );
    }
}

inline
uint32_t StixelsTracker::compute_maximum_pixelwise_motion_for_stixel( const Stixel& stixel )
{
    return m_maximal_pixelwise_motion_by_disp(stixel.disparity, 0);
}

void StixelsTracker::estimate_stixel_direction()
{
    for (uint32_t i = 0; i < m_tracker.size(); i++) {
        Stixel3d & stixel = m_tracker[i][m_tracker[i].size() - 1];
        
        stixel.direction = cv::Vec2d(0.0f, 0.0f);
        uint32_t numVectors = 0;
        for (uint32_t y = stixel.top_y; y <= stixel.bottom_y; y++) {
            const cv::Point2i currPoint(stixel.x, y);
            const cv::Point2i prevPoint = mp_denseTracker->getPrevPoint(currPoint);
            
            if (prevPoint != cv::Point2i(-1, -1)) {
                stixel.direction += cv::Vec2d(prevPoint.x - currPoint.x, prevPoint.y - currPoint.y);
                numVectors++;
            }
        }
        
        stixel.direction /= (double)numVectors;
//         stixel.direction /= cv::norm(stixel.direction);
        cout << stixel.direction << endl;
    }
}

void StixelsTracker::compute_static_stixels()
{
    
    cv::Mat mapXprev, mapYprev;
    mp_polarCalibration->getInverseMaps(mapXprev, mapYprev, 1);
    cv::Mat currPolar2LinearX, currPolar2LinearY;
    mp_polarCalibration->getMaps(currPolar2LinearX, currPolar2LinearY, 2);
    
    // Rectified difference is obtained
    cv::Mat diffRect;
    {
        cv::Mat polar1, polar2, diffPolar;
        mp_polarCalibration->getStoredRectifiedImages(polar1, polar2);
        cv::Mat polar1gray(polar1.size(), CV_8UC1);
        cv::Mat polar2gray(polar1.size(), CV_8UC1);
        cv::cvtColor(polar1, polar1gray, CV_BGR2GRAY);
        cv::cvtColor(polar2, polar2gray, CV_BGR2GRAY);
        cv::absdiff(polar1gray, polar2gray, diffPolar);
        
        cv::Mat inverseX, inverseY;
        mp_polarCalibration->getInverseMaps(inverseX, inverseY, 1);
        cv::remap(diffPolar, diffRect, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_CONSTANT);
    }
//     cv::threshold(diffRect, diffRect, 30, 255, cv::THRESH_BINARY);
    
    cv::Mat diffRectColor(diffRect.size(), CV_8UC3);
    cv::cvtColor(diffRect, diffRectColor, CV_GRAY2BGR);
    cv::Mat diffRectColorBig;
    cv::resize(diffRectColor, diffRectColorBig, cv::Size(1920, 1200));
    
    for (stixels_t::iterator it = current_stixels_p->begin(), it2 = previous_stixels_p->begin(); 
                    it != current_stixels_p->end(); it++, it2++) {
        
        double totalDiffs = 0.0f;
        {
            for (uint32_t j = min(it->bottom_y, it2->bottom_y); j <= max(it->bottom_y, it2->bottom_y); j++) {
                if (diffRect.at<uint8_t>(j, it->x) == 255)
                    totalDiffs++;
            }
            totalDiffs /= fabs(it->bottom_y - it->bottom_y) + 1;
        }
        
        //         cv::line(diffRectColor, cv::Point2d(it->x, it->bottom_y), cv::Point2d(it2->x, it2->bottom_y), cv::Scalar(255 - 255 * totalDiffs, 0, 255 * totalDiffs));
        //         cv::circle(diffRectColor, cv::Point2d(it->x, it->bottom_y), 1, cv::Scalar(255 - 255 * totalDiffs, 0, 255 * totalDiffs), -1);
        //         cv::circle(diffRectColor, cv::Point2d(it2->x, it2->bottom_y), 1, cv::Scalar(255 - 255 * totalDiffs, 0, 255 * totalDiffs), -1);
    }
        
//     for (stixels_t::iterator it = current_stixels_p->begin(), it2 = previous_stixels_p->begin(); 
//              it != current_stixels_p->end(); it++, it2++) {
// //         const cv::Point2d currPoint(it->x, it->bottom_y);
//         const cv::Point2d lastPoint(it2->x, it2->bottom_y);
//         const cv::Point2d & lastPointNow = get_polar_point(mapXprev, mapYprev, currPolar2LinearX, currPolar2LinearY, *it2);
//         cv::Point2d currPoint(-1, -1);
//         if (lastPointNow != currPoint)
//             currPoint = cv::Point2d(current_stixels_p->at(lastPointNow.x).x, current_stixels_p->at(lastPointNow.x).bottom_y);
    
    stixels_motion_t corresp = stixels_motion;

    for (uint32_t prevPos = 0; prevPos < previous_stixels_p->size(); prevPos++) {
        
        uint32_t currPos = 0;
        for (; currPos < corresp.size(); currPos++)
            if (corresp[currPos] == prevPos)
                break;
            
        cv::Point2d currPoint(-1, -1);
        if (currPos != corresp.size())
            currPoint = cv::Point2d(current_stixels_p->at(currPos).x, current_stixels_p->at(currPos).bottom_y);
        
        const cv::Point2d lastPoint(previous_stixels_p->at(prevPos).x, previous_stixels_p->at(prevPos).bottom_y);
        const cv::Point2d & lastPointNow = get_polar_point(mapXprev, mapYprev, currPolar2LinearX, currPolar2LinearY, previous_stixels_p->at(prevPos));

        cv::circle(diffRectColor, currPoint, 1, cv::Scalar(0, 0, 255), -1);
        cv::circle(diffRectColor, lastPointNow, 1, cv::Scalar(255, 0, 0), -1);
        cv::circle(diffRectColor, lastPoint, 1, cv::Scalar(0, 255, 0), -1);
        
        const float factorX = (float)diffRectColorBig.cols / (float)diffRectColor.cols;
        const float factorY = (float)diffRectColorBig.rows / (float)diffRectColor.rows;
        const cv::Point2d currPointBig(currPoint.x * factorX, currPoint.y * factorY);
        const cv::Point2d lastPointBig(lastPoint.x * factorX, lastPoint.y * factorY);
        const cv::Point2d lastPointNowBig(lastPointNow.x * factorX, lastPointNow.y * factorY);
        
        cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        if ((currPoint != cv::Point2d(-1, -1)) && (lastPointNow != cv::Point2d(-1, -1)))
            cv::line(diffRectColorBig, currPointBig, lastPointNowBig, color);
        if ((lastPointNow != cv::Point2d(-1, -1)) && (lastPoint != cv::Point2d(-1, -1)))
            cv::line(diffRectColorBig, lastPointNowBig, lastPointBig, color);
        
        cv::circle(diffRectColorBig, currPointBig, 1, cv::Scalar(0, 0, 255), -1);
        cv::circle(diffRectColorBig, lastPointNowBig, 1, cv::Scalar(255, 0, 0), -1);
        cv::circle(diffRectColorBig, lastPointBig, 1, cv::Scalar(0, 255, 0), -1);
    }
    
    cv::imshow("Thresh1", diffRectColor);
    cv::imshow("polarTrack", diffRectColorBig);
}

float StixelsTracker::compute_dense_tracking_score(const Stixel& currStixel, const Stixel& prevStixel)
{

    float matched = 0.0f, unmatched = 0.0f;
    for (uint32_t y = currStixel.top_y; y <= currStixel.bottom_y; y++) {
        const cv::Point2i currPoint(currStixel.x, y);
        const cv::Point2i prevPoint = mp_denseTracker->getPrevPoint(currPoint);
        
        if (prevPoint != cv::Point2i(-1, -1)) {
            if (prevPoint.x == prevStixel.x)
                matched += 1.0f;
            else
                unmatched += 1.0f;
        }
    }
    
//     if ((matched + unmatched) == 0)
//         return 0;
//     else
//         return 255 * matched / (matched + unmatched);
    return matched;
}

float StixelsTracker::compute_polar_SAD(const Stixel& stixel1, const Stixel& stixel2)
{
    
    cv::Point2d point1(stixel1.x, 0.0f);
    cv::Point2d point2(stixel2.x, 0.0f);
    
    const double height1 = stixel1.bottom_y - stixel1.top_y; 
    const double height2 = stixel2.bottom_y - stixel2.top_y;
    const double height = max(height1, height2);
    
    const double factor1 = height1 / height;
    const double factor2 = height2 / height;
    
    cv::Mat polarImg1, polarImg2;
    mp_polarCalibration->getStoredRectifiedImages(polarImg1, polarImg2);
    
    cv::Mat mapXprev, mapYprev, mapXcurr, mapYcurr;
    mp_polarCalibration->getInverseMaps(mapXprev, mapYprev, 1);
    mp_polarCalibration->getInverseMaps(mapXcurr, mapYcurr, 2);
    
    float sad = 0.0;
    double validPoints = 0.0f;

    for (uint32_t i = 0; i <= height; i++) {
        const cv::Point2d pos1 = cv::Point2d(stixel1.x, stixel1.top_y + factor1 * i);
        const cv::Point2d pos2 = cv::Point2d(stixel2.x, stixel2.top_y + factor2 * i);
    
        cv::Point2d p1, p2;
        p1 = get_polar_point(mapXprev, mapYprev, pos1);
        p2 = get_polar_point(mapXcurr, mapYcurr, pos2);
        
        if ((p1 == cv::Point2d(-1, -1)) || (p2 == cv::Point2d(-1, -1)))
            continue;
        
        validPoints += 1.0f;
            
        const cv::Vec3b & px1 = polarImg2.at<cv::Vec3b>(p1.y, p1.x);
        const cv::Vec3b & px2 = polarImg1.at<cv::Vec3b>(p2.y, p2.x);
        
        const cv::Vec3b diffPx = px1 - px2;
        sad += fabs(cv::sum(diffPx)[0]);
    }
    
    return sad / validPoints / polarImg1.channels();
}

float StixelsTracker::compute_polar_SAD(const Stixel& stixel1, const Stixel& stixel2,
                                        const input_image_const_view_t& image_view1, const input_image_const_view_t& image_view2,
                                        const unsigned int stixel_horizontal_padding)
{
    const unsigned int stixel_representation_width = stixel1.width + 2 * stixel_horizontal_padding;
    
    const unsigned int number_of_channels = image_view1.num_channels();
    
    stixel_representation_t stixel_representation1;
    stixel_representation_t stixel_representation2;
    
    compute_stixel_representation_polar( stixel1, image_view1, stixel_representation1, stixel_horizontal_padding, m_mapXcurr, m_mapYcurr, m_polarImg2 );    
    compute_stixel_representation_polar( stixel2, image_view2, stixel_representation2, stixel_horizontal_padding, m_mapXprev, m_mapYprev, m_polarImg1 );
    
    float pixelwise_sad = 0;
    
    for( unsigned int c = 0; c < number_of_channels; ++c )
    {
        const Eigen::MatrixXf& current_stixel_representation_channel = stixel_representation1[ c ];
        const Eigen::MatrixXf& previous_stixel_representation_channel = stixel_representation2[ c ];
        
        for( unsigned int y = 0; y < stixel_representation_height; ++y )
        {
            for( unsigned int x = 0; x < stixel_representation_width; ++x )
            {
                pixelwise_sad += fabs( current_stixel_representation_channel( y, x ) - previous_stixel_representation_channel( y, x ) );
                
            } // End of for( x )
            
        } // End of for( y )
        
    } // End of for( c )
    
    pixelwise_sad = pixelwise_sad / number_of_channels;
    pixelwise_sad = pixelwise_sad / ( stixel_representation_height * stixel_representation_width );
    
    stixel_representation1.clear();
    stixel_representation2.clear();
    
    return pixelwise_sad;
}

void StixelsTracker::compute_stixel_representation_polar( const Stixel &stixel, const input_image_const_view_t& image_view_hosting_the_stixel,
                                                          stixel_representation_t &stixel_representation, const unsigned int stixel_horizontal_padding,
                                                          const cv::Mat & mapX, const cv::Mat & mapY, const cv::Mat & polarImg )
{
    const unsigned int stixel_representation_width = stixel.width + 2 * stixel_horizontal_padding;    
    
    const int stixel_height = abs( stixel.top_y - stixel.bottom_y );
    const int stixel_effective_part_height = stixel_height;
    
    const float reduction_ratio = float( stixel_representation_height ) / float( stixel_effective_part_height );
    
    // Image boundary conditions are NOT checked for speed efficiency !
    if( (stixel.width % 2) != 1 ) {
        printf("stixel.width == %i\n", stixel.width);
        throw std::invalid_argument( "DummyStixelMotionEstimator::compute_stixel_representation() -- The width of stixel should be an odd number !" );
    }
    
    const int32_t minX = stixel.x - ( stixel.width - 1 ) / 2 - stixel_horizontal_padding;
    const int32_t maxX = stixel.x + ( stixel.width - 1 ) / 2 + stixel_horizontal_padding;
    
//     if( minX < 0 || maxX >= image_view_hosting_the_stixel.width() ) {
    if( stixel.x - ( stixel.width - 1 ) / 2 - stixel_horizontal_padding < 0 ||
        stixel.x + ( stixel.width - 1 ) / 2 + stixel_horizontal_padding >= image_view_hosting_the_stixel.width() )
    {
        
        throw std::invalid_argument( "DummyStixelMotionEstimator::compute_stixel_representation() -- The stixel representation should obey the image boundaries !" );
    }    
    
    const unsigned int number_of_channels = image_view_hosting_the_stixel.num_channels();
    
    stixel_representation.clear();
    stixel_representation.resize( number_of_channels );   
    
//     cv::Mat dbgImg = cv::Mat::zeros(stixel_representation_height, stixel_representation_width, CV_8UC3);
    
    for( unsigned int c = 0; c < number_of_channels; ++c ) {
        stixel_representation[ c ].resize( stixel_representation_height, stixel_representation_width );
        
    } // End of for( c )
    
    for( unsigned int y = 0; y < stixel_representation_height; ++y ) {
        const float projected_y = float( y ) / reduction_ratio;
        
        const float projected_upper_y = std::ceil( projected_y );
        const float projected_lower_y = std::floor( projected_y );
        
        // The coefficients are in reverse order (sum of coefficients is 1)
        float coefficient_lower_y = projected_upper_y - projected_y;
        float coefficient_upper_y = projected_y - projected_lower_y;
        
        // If the projected pixel falls just on top of an integer coordinate
        if( coefficient_lower_y + coefficient_upper_y < 0.05 ) {
            coefficient_lower_y = 0.5;
            coefficient_upper_y = 0.5;
        }
                
        for( unsigned int x = 0; x < stixel_representation_width; ++x ) {
            cv::Point2d polarLower = get_polar_point(mapX, mapY, cv::Point2d(x, projected_lower_y));
            cv::Point2d polarUpper = get_polar_point(mapX, mapY, cv::Point2d(x, projected_upper_y));
                        
            if ((polarLower != cv::Point2d(-1, -1)) && (polarUpper != cv::Point2d(-1, -1))) {
                    
                const cv::Vec3b & pxLower = polarImg.at<cv::Vec3b>(polarLower.y, polarLower.x);
                const cv::Vec3b & pxUpper = polarImg.at<cv::Vec3b>(polarLower.y, polarLower.x);
                
                for( unsigned int c = 0; c < number_of_channels; ++c ) {
                    ( stixel_representation[ c ] )( y, x ) = coefficient_lower_y * pxLower[ c ] + coefficient_upper_y * pxUpper[ c ];
                } // End of for( c )
            }
            
        } // End of for( x )
        
    } // End of for( y )
    
//     for( unsigned int y = stixel.top_y, representationY = 0; y < stixel.bottom_y; ++y, representationY++) {
// 
//         
//         
//         float projectedY = reduction_ratio * representationY; 
//         
//         for( unsigned int x = minX, representationX = 0; x <= maxX; ++x, representationX++ ) {
//             
//         }
//     }

//     for( unsigned int y = stixel.top_y, representationY = 0; y < stixel.bottom_y; ++y, representationY++ ) {
//         
//         const float reduced_Y = float( representationY ) / reduction_ratio;
// 
//         for( unsigned int x = minX, representationX = 0; x <= maxX; ++x, representationX++ ) {
//         
//             cv::Point2d p = get_polar_point(mapX, mapY, cv::Point2d(x, y));
//             
//             if (p != cv::Point2d(-1, -1)) {
//         
//                 const float projected_x = p.x;
//                 const float projected_y = p.y;
//                 
//                 const float projected_upper_y = std::ceil( projected_y );
//                 const float projected_lower_y = std::floor( projected_y );
//             
//                 // The coefficients are in reverse order (sum of coefficients is 1)
//                 float coefficient_lower_y = projected_upper_y - projected_y;
//                 float coefficient_upper_y = projected_y - projected_lower_y;
//                 
//                 // If the projected pixel falls just on top of an integer coordinate
//                 if( coefficient_lower_y + coefficient_upper_y < 0.05 ) {
//                     coefficient_lower_y = 0.5;
//                     coefficient_upper_y = 0.5;
//                 }
// 
//                 const cv::Vec3b & pxLower = m_polarImg2.at<cv::Vec3b>(projected_lower_y, projected_x);
//                 const cv::Vec3b & pxUpper = m_polarImg1.at<cv::Vec3b>(projected_upper_y, projected_x);
//                 
//                 for( unsigned int c = 0; c < number_of_channels; ++c ) {
//                     cout << "c " << c << endl;
//                     cout << "number_of_channels " << number_of_channels << endl;
//                     cout << "reduced_Y " << reduced_Y << endl;
//                     cout << "representationX " << representationX << endl;
//                     cout << "pxLower " << pxLower << endl;
//                     cout << "pxUpper " << pxUpper << endl;
//                     cout << "coefficient_lower_y " << representationX << endl;
//                     cout << "coefficient_upper_y " << representationX << endl;
//                     cout << "( stixel_representation[ c ] )( reduced_Y, representationX ) " << ( stixel_representation[ c ] )( (unsigned int)reduced_Y, representationX ) << endl;
//                     ( stixel_representation[ c ] )( (unsigned int)reduced_Y, representationX ) = 0; //coefficient_lower_y * pxLower[c] + coefficient_upper_y * pxUpper[c];
// //                     dbgImg.at<cv::Vec3b>(reduced_Y, representationX)[c] = coefficient_lower_y * pxLower[c] + coefficient_upper_y * pxUpper[c];
//                 } // End of for( c )
//             }
//         } // End of for( x )
//     } // End of for( y )
    
//     cv::imshow("dbgImg", dbgImg);
//     cv::waitKey(0);
    
    return;
}

void StixelsTracker::draw_polar_SAD(cv::Mat& img, const Stixel& stixel1, const Stixel& stixel2)
{
    cv::Point2d point1(stixel1.x, 0.0f);
    cv::Point2d point2(stixel2.x, 0.0f);
    
    const double height1 = stixel1.bottom_y - stixel1.top_y; 
    const double height2 = stixel2.bottom_y - stixel2.top_y;
    const double height = max(height1, height2);
    
    const double factor1 = height1 / height;
    const double factor2 = height2 / height;
    
    cv::Mat polarImg1, polarImg2;
    mp_polarCalibration->getStoredRectifiedImages(polarImg1, polarImg2);
    
    cv::Mat mapXprev, mapYprev, mapXcurr, mapYcurr;
    mp_polarCalibration->getInverseMaps(mapXprev, mapYprev, 1);
    mp_polarCalibration->getInverseMaps(mapXcurr, mapYcurr, 2);
    
    for (uint32_t i = 0; i <= height; i++) {
        const cv::Point2d pos1 = cv::Point2d(stixel1.x, stixel1.top_y + factor1 * i);
        const cv::Point2d pos2 = cv::Point2d(stixel2.x, stixel2.top_y + factor2 * i);
        
        cv::Point2d p1, p2;
        p1 = get_polar_point(mapXprev, mapYprev, pos1);
        p2 = get_polar_point(mapXcurr, mapYcurr, pos2);
        
        if ((p1 == cv::Point2d(-1, -1)) || (p2 == cv::Point2d(-1, -1))) {
            img.at<cv::Vec3b>(pos1.y, pos1.x)= cv::Vec3b::all(0);
        } else {
            const cv::Vec3b & px1 = polarImg1.at<cv::Vec3b>(p1.y, p1.x);
            const cv::Vec3b & px2 = polarImg2.at<cv::Vec3b>(p2.y, p2.x);
            
            const cv::Vec3b diffPx = px1 - px2;
            double sad = fabs(cv::sum(diffPx)[0]) / 3.0;
            
            img.at<cv::Vec3b>(pos1.y, pos1.x)= cv::Vec3b::all(sad);
        }
    }
}

void StixelsTracker::computeMotionWithGraphsAndHistogram()
{
    lemon::SmartGraph graph;
    lemon::SmartGraph::EdgeMap <float> costs(graph);
    lemon::SmartGraph::NodeMap <uint32_t> nodeIdx(graph);
    graph.reserveNode(current_stixels_p->size() + previous_stixels_p->size());
    graph.reserveEdge(current_stixels_p->size() * previous_stixels_p->size());
    
    BOOST_FOREACH (const Stixel & stixel, *previous_stixels_p)
    nodeIdx[graph.addNode()] = stixel.x;
    BOOST_FOREACH (const Stixel & stixel, *current_stixels_p)
    nodeIdx[graph.addNode()] = stixel.x;
    
    cv::MatND hist1, hist2;
    cv::Mat lastImg = m_currImg;
    if (lastImg.empty())
        gil2opencv(previous_image_view, lastImg);
    gil2opencv(current_image_view, m_currImg);
    BOOST_FOREACH (const Stixel & currStixel, *current_stixels_p) {
        
        const uint32_t & maxMotionForStixel = compute_maximum_pixelwise_motion_for_stixel( currStixel );
        
        computeHistogram(hist1, m_currImg, currStixel);
        
        for (uint32_t stixelIdx = max((int)0, (int)(currStixel.x - maxMotionForStixel)); 
             stixelIdx <= min((int)(previous_stixels_p->size() - 1), (int)(currStixel.x + maxMotionForStixel)); stixelIdx++) {
         
            const Stixel & prevStixel = previous_stixels_p->at(stixelIdx);
        
            computeHistogram(hist2, lastImg, prevStixel);
            
    //             cout << "Comparing " << prevStixel.x << " and " << currStixel.x << endl;
            const int32_t pixelwise_motion = prevStixel.x - currStixel.x;
            const int32_t pixelwise_motionY = fabs(prevStixel.bottom_y - currStixel.bottom_y);
        
            if( pixelwise_motion >= -( int( maxMotionForStixel ) ) &&
                pixelwise_motion <= int( maxMotionForStixel ) &&
                pixelwise_motionY <= int( maxMotionForStixel ) &&
                ( currStixel.type != Stixel::Occluded && prevStixel.type != Stixel::Occluded )) {
                    
                float histogramComparisonScore = 1.0 - compareHistogram(hist1, hist2, currStixel, prevStixel);
                
                const lemon::SmartGraph::Edge & e = graph.addEdge(graph.nodeFromId(prevStixel.x), graph.nodeFromId(currStixel.x + previous_stixels_p->size()));
                costs[e] = histogramComparisonScore;
                
            }
        }
    }
    
//     const float maxCost = motion_cost_matrix.maxCoeff();
//     for (uint32_t prevIdx = 0; prevIdx < previous_stixels_p->size(); prevIdx++) {
//         const Stixel & prevStixel = previous_stixels_p->at(prevIdx);
//         for (uint32_t currIdx = 0; currIdx < current_stixels_p->size(); currIdx++) {
//             const Stixel & currStixel = current_stixels_p->at(currIdx);
//             
//             const int32_t pixelwise_motion = prevStixel.x - currStixel.x;
//             const uint32_t & maximum_motion_in_pixels_for_current_stixel = compute_maximum_pixelwise_motion_for_stixel( currStixel );
//             
//             const int32_t pixelwise_motionY = fabs(prevStixel.bottom_y - currStixel.bottom_y);
//             
//             const uint32_t rowIndex = pixelwise_motion + maximum_possible_motion_in_pixels;
//             
//             if( pixelwise_motion >= -( int( maximum_motion_in_pixels_for_current_stixel ) ) &&
//                 pixelwise_motion <= int( maximum_motion_in_pixels_for_current_stixel ) &&
//                 pixelwise_motionY <= int( maximum_motion_in_pixels_for_current_stixel ) &&
//                 (motion_cost_assignment_matrix(rowIndex, currIdx))) {
//                 
//                 const float & polarDist = m_stixelsPolarDistMatrix(rowIndex, currIdx);
//             
// //                 if (polarDist > 1.0f) {
//                     const float & cost = maxCost - motion_cost_matrix(rowIndex, currIdx);
//                     
//                     const lemon::SmartGraph::Edge & e = graph.addEdge(graph.nodeFromId(prevIdx), graph.nodeFromId(currIdx + previous_stixels_p->size()));
//                     costs[e] = cost;
// //                 }
//             }
//         }
//     }
    
    lemon::MaxWeightedMatching< lemon::SmartGraph, lemon::SmartGraph::EdgeMap <float> > graphMatcher(graph, costs);
    
    graphMatcher.run();
    
    const lemon::SmartGraph::NodeMap<lemon::SmartGraph::Arc> & matchingMap = graphMatcher.matchingMap();
    
    stixels_motion = vector<int>(current_stixels_p->size(), -1);
    for (uint32_t i = 0; i < previous_stixels_p->size(); i++) {
        if (graphMatcher.mate(graph.nodeFromId(i)) != lemon::INVALID) {
            lemon::SmartGraph::Arc arc = matchingMap[graph.nodeFromId(i)];
            if ((graph.id(graph.target(arc)) - previous_stixels_p->size()) != graph.id(graph.source(arc)))
                stixels_motion[graph.id(graph.target(arc)) - previous_stixels_p->size()] = graph.id(graph.source(arc));
        }
    }
}

void StixelsTracker::computeMotionWithGraphs()
{
    lemon::SmartGraph graph;
    lemon::SmartGraph::EdgeMap <float> costs(graph);
    lemon::SmartGraph::NodeMap <uint32_t> nodeIdx(graph);
    graph.reserveNode(current_stixels_p->size() + previous_stixels_p->size());
    graph.reserveEdge(current_stixels_p->size() * previous_stixels_p->size());
    
    BOOST_FOREACH (const Stixel & stixel, *previous_stixels_p)
        nodeIdx[graph.addNode()] = stixel.x;
    BOOST_FOREACH (const Stixel & stixel, *current_stixels_p)
        nodeIdx[graph.addNode()] = stixel.x;
    
    const float maxCost = motion_cost_matrix.maxCoeff();
    for (uint32_t prevIdx = 0; prevIdx < previous_stixels_p->size(); prevIdx++) {
        const Stixel & prevStixel = previous_stixels_p->at(prevIdx);
                
        for (uint32_t currIdx = 0; currIdx < current_stixels_p->size(); currIdx++) {
            const Stixel & currStixel = current_stixels_p->at(currIdx);
            
            const int32_t pixelwise_motion = prevStixel.x - currStixel.x;
            const uint32_t & maximum_motion_in_pixels_for_current_stixel = compute_maximum_pixelwise_motion_for_stixel( currStixel );
            
            const int32_t pixelwise_motionY = fabs(prevStixel.bottom_y - currStixel.bottom_y);
            
            const uint32_t rowIndex = pixelwise_motion + maximum_possible_motion_in_pixels;
            
            if( pixelwise_motion >= -( int( maximum_motion_in_pixels_for_current_stixel ) ) &&
                pixelwise_motion <= int( maximum_motion_in_pixels_for_current_stixel ) &&
                pixelwise_motionY <= int( maximum_motion_in_pixels_for_current_stixel ) &&
                (motion_cost_assignment_matrix(rowIndex, currIdx))) {
                    
                const float & polarDist = m_stixelsPolarDistMatrix(rowIndex, currIdx);
                
//                 if (polarDist > 1.0f) {
                    const float & cost = maxCost - motion_cost_matrix(rowIndex, currIdx);
                    
                    const lemon::SmartGraph::Edge & e = graph.addEdge(graph.nodeFromId(prevIdx), graph.nodeFromId(currIdx + previous_stixels_p->size()));
                    costs[e] = cost;
//                 }
            }
        }
    }
    
    lemon::MaxWeightedMatching< lemon::SmartGraph, lemon::SmartGraph::EdgeMap <float> > graphMatcher(graph, costs);
    
    graphMatcher.run();
    
    const lemon::SmartGraph::NodeMap<lemon::SmartGraph::Arc> & matchingMap = graphMatcher.matchingMap();
    
    stixels_motion = vector<int>(current_stixels_p->size(), -1);
    for (uint32_t i = 0; i < previous_stixels_p->size(); i++) {
        if (graphMatcher.mate(graph.nodeFromId(i)) != lemon::INVALID) {
            lemon::SmartGraph::Arc arc = matchingMap[graph.nodeFromId(i)];
            if ((graph.id(graph.target(arc)) - previous_stixels_p->size()) != graph.id(graph.source(arc)))
                stixels_motion[graph.id(graph.target(arc)) - previous_stixels_p->size()] = graph.id(graph.source(arc));
        }
    }
}

void StixelsTracker::updateTracker()
{
    const stixels_t * currStixels = current_stixels_p;
    stixels_motion_t corresp = stixels_motion;
    
    if (m_tracker.size() == 0) {
        stixels3d_t newStixels3d;
        newStixels3d.reserve(currStixels->size());
        m_tracker.resize(currStixels->size());
        for (uint32_t i = 0; i < currStixels->size(); i++) {
            Stixel3d currStixel3d(currStixels->at(i));
            currStixel3d.update3dcoords(stereo_camera);
            currStixel3d.isStatic = false;
            
            m_tracker[i].push_back(currStixel3d);
            
            currStixel3d.valid_forward_delta_x = false;
            newStixels3d.push_back(currStixel3d);
        }
        m_stixelsHistoric.push_front(newStixels3d);
        return;
    }
    
    if (m_stixelsHistoric.size() > MAX_ITERATIONS_STORED) {
        m_stixelsHistoric.pop_back();
    }
    
    t_tracker tmpTracker(m_tracker.size());
    copy(m_tracker.begin(), m_tracker.end(), tmpTracker.begin());
    m_tracker.clear();
    m_tracker.resize(currStixels->size());
    
    stixels3d_t & lastStixels3d = m_stixelsHistoric[0];
    stixels3d_t newStixels3d;
    newStixels3d.reserve(currStixels->size());
    
    for (uint32_t i = 0; i < currStixels->size(); i++) {
        if (corresp[i] >= 0) {
            m_tracker[i] = tmpTracker[corresp[i]];
        }
        Stixel3d currStixel3d(currStixels->at(i));
        currStixel3d.update3dcoords(stereo_camera);
        
//         currStixel3d.isStatic = false;
//         if ((corresp[i] >= 0) && (compute_polar_SAD(currStixels->at(i), previous_stixels_p->at(corresp[i])) < m_minPolarSADForBeingStatic))
//             currStixel3d.isStatic = true;

        currStixel3d.valid_backward_delta_x = false;
        currStixel3d.valid_forward_delta_x = false;
        if (corresp[i] >= 0) {
            lastStixels3d[corresp[i]].forward_delta_x = i;
            lastStixels3d[corresp[i]].valid_forward_delta_x = true;
            
            currStixel3d.backward_delta_x = corresp[i];
            currStixel3d.valid_backward_delta_x = true;
        }
        
        m_tracker[i].push_back(currStixel3d);
        newStixels3d.push_back(currStixel3d);
    }
    m_stixelsHistoric.push_front(newStixels3d);
}

void StixelsTracker::getClusters()
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
    cloud->reserve(current_stixels_p->size());
    
    for (uint32_t i = 0; i < m_tracker.size(); i++) {
        pcl::PointXYZL pointPCL;
        if (stixels_motion[i] >= 0) {
            const cv::Point3d & point = m_tracker[i][m_tracker[i].size() - 1].bottom3d;
            pointPCL.x = point.x;
            pointPCL.y = 0.0f; //point.y;
            pointPCL.z = point.z;
            pointPCL.label = 1;
        }
        cloud->push_back(pointPCL);
    }
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZL>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZL>);
    tree->setInputCloud (cloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZL> ec;
    ec.setClusterTolerance (m_minDistBetweenClusters);
    ec.setMinClusterSize (3); 
    ec.setMaxClusterSize (m_tracker.size());
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    m_clusters.clear();
    m_clusters.resize(m_tracker.size());

    m_objects.clear();
    m_objects.reserve(cluster_indices.size());
    
    uint32_t clusterIdx = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it, ++clusterIdx)
    {
        const int32_t & idxBegin = it->indices[0];
        const int32_t & idxEnd = it->indices[it->indices.size() - 1];
        const Stixel3d & stixelBegin = m_tracker[idxBegin][m_tracker[idxBegin].size() - 1];
        const Stixel3d & stixelEnd = m_tracker[idxEnd][m_tracker[idxEnd].size() - 1];
        
        const double clusterWidth = stixelEnd.bottom3d.x - stixelBegin.bottom3d.x; 

        uint32_t trackLenght = 0;
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++) {
            if ((cloud->at(*pit).label == 0) || (clusterWidth < m_minAllowedObjectWidth)) {
                m_clusters[*pit] = -1;
            } else {
                m_clusters[*pit] = clusterIdx;
                if (m_tracker[*pit].size() > trackLenght)
                    trackLenght = m_tracker[*pit].size();
            }
        }

        if ((clusterWidth > m_minAllowedObjectWidth) && 
            (cloud->at(it->indices[0]).label != 0) && (trackLenght > 2)) {
            
            vector <int> object(it->indices.size());
            copy(it->indices.begin(), it->indices.end(), object.begin());
            m_objects.push_back(object);
        }
    }
}

void StixelsTracker::projectPointInTopView(const cv::Point3d & point3d, const cv::Mat & imgTop, cv::Point2d & point2d)
{
    const double maxDistZ = 20.0;
    const double maxDistX = maxDistZ / 2.0;
    
    // v axis corresponds to Z
    point2d.y = imgTop.rows - ((imgTop.rows - 10) * min(maxDistZ, point3d.z) / maxDistZ);
    
    // u axis corresponds to X
    point2d.x = ((imgTop.cols / 2.0) * min(maxDistX, point3d.x) / maxDistX) + imgTop.cols / 2;
}



void StixelsTracker::drawTracker(cv::Mat& img, cv::Mat & imgTop)
{
    
    if (m_color.size() == 0) {
        m_color.resize(current_stixels_p->size());
        
        uint32_t division = m_color.size() / 3;
        for (uint32_t i = 1; i <= m_color.size(); i++) {
            m_color[i - 1] = cv::Scalar((i * 50) % 256, (i * 100) % 256, (i * 200) % 256);
        }
        
    }
    
    gil2opencv(current_image_view, img);
    imgTop = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    
    cv::rectangle(img, cv::Point2d(0, 0), cv::Point2d(img.cols - 1, 20), cv::Scalar::all(0), -1);
    
    stringstream oss;
    oss << "SAD = " << m_sad_factor << ", Height = " << m_height_factor << 
           ", Polar distance = " << m_polar_dist_factor << ", Polar SAD = " << m_polar_sad_factor <<
           ", Dense Tracking = " << m_dense_tracking_factor;
    cv::putText(img, oss.str(), cv::Point2d(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(255));
    cv::putText(imgTop, oss.str(), cv::Point2d(2, 7), cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar::all(255));
    
    if (0) {
        uint32_t clusterIdx = 0;
        for (vector < vector <int> >::iterator it = m_objects.begin(); it != m_objects.end(); it++, clusterIdx++) {
            const cv::Scalar & color =  m_color[clusterIdx];
            
            cv::Point2d corner1(current_stixels_p->at(it->at(0)).x, current_stixels_p->at(it->at(0)).bottom_y);
            cv::Point2d corner2(current_stixels_p->at(it->at(it->size() - 1)).x, current_stixels_p->at(it->at(it->size() - 1)).top_y);
            
            for (vector<int>::iterator it2 = it->begin(); it2 != it->end(); it2++) {
                
                stixels3d_t & track = m_tracker[*it2];
                stixels3d_t::iterator itTrack = track.begin();
                itTrack++;
                for (; itTrack != track.end(); itTrack++) {
                    cv::line(img, itTrack->getBottom2d<cv::Point2d>(), (itTrack - 1)->getBottom2d<cv::Point2d>(), color);
                    
                    cv::Point2d p1Top, p2Top;
                    projectPointInTopView(itTrack->bottom3d, imgTop, p1Top);
                    projectPointInTopView((itTrack - 1)->bottom3d, imgTop, p2Top);
                    cv::line(imgTop, p1Top, p2Top, color);
                }
                
                if (current_stixels_p->at(*it2).bottom_y > corner1.y) 
                    corner1.y = current_stixels_p->at(*it2).bottom_y;
                if (current_stixels_p->at(*it2).top_y < corner2.y) 
                    corner2.y = current_stixels_p->at(*it2).top_y;
            }
            
            cv::rectangle(img, corner1, corner2, color);
        }
    } else if (1) {
        for (vector < stixels3d_t >::iterator it = m_tracker.begin(); it != m_tracker.end(); it++) {
            const cv::Scalar & color =  m_color[it->begin()->x];
//             const cv::Scalar color = /*(it->at(it->size() - 1).isStatic)? cv::Scalar(255, 0, 0) : */cv::Scalar(0, 0, 255);
            for (stixels3d_t::iterator it2 = it->begin() + 1; it2 != it->end(); it2++) {
                const cv::Point2d & p1 = (it2 - 1)->getBottom2d<cv::Point2d>();
                const cv::Point2d & p2 = it2->getBottom2d<cv::Point2d>();
                
//                 draw_polar_SAD(img, *(it2 - 1), *it2);
//                 float sad = compute_polar_SAD(*(it2 - 1), *it2);
//                 const cv::Point2d p2c(p2.x, it2->top_y);
//                 const cv::Point2d p2d(p2.x, it2->top_y - 20);
//                 const cv::Point2d p2e(p2.x, it2->top_y - 40);
// //                 cv::line(img, p2c, p2d, cv::Scalar(0, 0, 255));
//                 cv::line(img, p2d, p2e, cv::Scalar(sad, sad, sad));
                
                cv::line(img, p1, p2, color);
            }
            
//             const cv::Point2d & lastPoint = it->at(it->size() - 1).getBottom2d<cv::Point2d>();
//             cv::circle(img, lastPoint, 3, color, -1);
        }
    } else {
        for (vector < stixels3d_t >::iterator it = m_tracker.begin(); it != m_tracker.end(); it++) {
            const cv::Scalar & color =  m_color[it->begin()->x];

            Stixel3d & stixel = it->at(it->size() - 1);
            const cv::Point2d & p1 = stixel.getBottom2d<cv::Point2d>();
            const cv::Point2d & p2 = p1 + 5 * cv::Point2d(stixel.direction);
            
            cv::circle(img, p1, 1, color, -1);
            cv::line(img, p1, p2, color);
        }
    }
    
    
    cv::rectangle(imgTop, cv::Point2d(0, 0), cv::Point2d(imgTop.cols - 1, imgTop.rows - 1), cv::Scalar::all(255));
}

void StixelsTracker::drawTracker(cv::Mat& img)
{
    cv::rectangle(img, cv::Point2d(0, 0), cv::Point2d(img.cols - 1, 20), cv::Scalar::all(0), -1);
    
    stringstream oss;
    oss << "SAD = " << m_sad_factor << ", Height = " << m_height_factor << 
    ", Polar distance = " << m_polar_dist_factor << ", Polar SAD = " << m_polar_sad_factor <<
    ", Dense Tracking = " << m_dense_tracking_factor;
    cv::putText(img, oss.str(), cv::Point2d(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(255));
    
    for (vector < stixels3d_t >::iterator it = m_tracker.begin(); it != m_tracker.end(); it++) {
        const cv::Scalar color = (it->at(it->size() - 1).isStatic)? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
        
        const cv::Point2d & lastPointB = it->at(it->size() - 1).getBottom2d<cv::Point2d>();
        const cv::Point2d & lastPointT = it->at(it->size() - 1).getTop2d<cv::Point2d>();
        cv::circle(img, lastPointB, 1, color, -1);
        cv::circle(img, lastPointT, 1, color, -1);
    }
    
    
}

void StixelsTracker::drawDenseTracker(cv::Mat& img)
{
    mp_denseTracker->drawTracks(img);
}

stixels3d_t StixelsTracker::getLastStixelsAfterTracking()
{
    stixels3d_t stixels;
    stixels.reserve(m_tracker.size());
    for (vector < stixels3d_t >::iterator it = m_tracker.begin(); it != m_tracker.end(); it++) {
        if (it->size() > 1)
            stixels.push_back(it->at(0));
    }

    return stixels;
}

void StixelsTracker::computeHistogram(cv::Mat& hist, const cv::Mat& img, const Stixel& stixel)
{
    cv::Rect roi = cv::Rect(stixel.x, stixel.top_y, 1, stixel.bottom_y - stixel.top_y + 1);
    
    cv::Mat roiImg = img(roi);
    
    const int histSize = 64;
    
    cv::Mat roiGray;
    cv::cvtColor(roiImg, roiGray, CV_BGR2GRAY);
//     cv::imshow("roiImg", roiImg);
//     roiImg.setTo(cv::Scalar(0, 0, 255));
//     cv::imshow("computeHistogram", img);
    
    cv::calcHist(&roiGray, 1, 0, cv::Mat(), hist, 1, &histSize, 0);
    normalize(hist, hist, 0, 255, CV_MINMAX, CV_32F);
    
//     // Visualization
//     cv::Mat histImage = cv::Mat::ones(200, 320, CV_8U) * 255;
// 
// 
//     histImage = cv::Scalar::all(255);
//     int binW = cvRound((double)histImage.cols/histSize);
// 
//     for( int i = 0; i < histSize; i++ ) {
//         cout << "hist[" << i << "] = " << hist.at<float>(i) << endl;
//         cv::rectangle( histImage, cv::Point(i*binW, histImage.rows),
//                     cv::Point((i+1)*binW, histImage.rows - cvRound(hist.at<float>(i))),
//                     cv::Scalar::all(0), -1, 8, 0 );
//     }
//     imshow("histogram", histImage);
    
//     cv::waitKey(0);
}

float StixelsTracker::compareHistogram(const cv::Mat& hist1, const cv::Mat& hist2, const Stixel& stixel1, const Stixel& stixel2) {
//     if (stixel1.x != 320)
//         return 0.0f;
//     
//     cv::Mat tmp1, tmp2;
//     gil2opencv(previous_image_view, tmp2);
//     m_currImg.copyTo(tmp1);
//     
//     cv::line(tmp1, cv::Point2d(stixel1.x, stixel1.top_y), 
//              cv::Point2d(stixel1.x, stixel1.bottom_y), cv::Scalar(0, 0, 255), 1);
//     cv::line(tmp2, cv::Point2d(stixel2.x, stixel2.top_y), 
//              cv::Point2d(stixel2.x, stixel2.bottom_y), cv::Scalar(0, 0, 255), 1);
//     
//     cv::imshow("currImg", tmp1);
//     cv::imshow("prevImg", tmp2);
// 
//     const int histSize = 64;
//     cv::Mat histImage1 = cv::Mat::ones(256, 320, CV_8U) * 255;
//     cv::Mat histImage2 = cv::Mat::ones(256, 320, CV_8U) * 255;
// 
//     histImage1 = cv::Scalar::all(255);
//     histImage2 = cv::Scalar::all(255);
// 
//     int binW = cvRound((double)histImage1.cols/histSize);
// 
//     for( int i = 0; i < histSize; i++ ) {
//         cv::rectangle( histImage1, cv::Point(i*binW, histImage1.rows),
//                     cv::Point((i+1)*binW, histImage1.rows - cvRound(hist1.at<float>(i))),
//                     cv::Scalar::all(0), -1, 8, 0 );
//         cv::rectangle( histImage2, cv::Point(i*binW, histImage2.rows),
//                        cv::Point((i+1)*binW, histImage2.rows - cvRound(hist2.at<float>(i))),
//                        cv::Scalar::all(0), -1, 8, 0 );
//     }
//     cv::imshow("histogramCurr", histImage1);
//     cv::imshow("histogramPrev", histImage2);
//     
//     cv::waitKey(0);
//     
//     float compCorrel = cv::compareHist(hist1, hist2, CV_COMP_CORREL);
//     float compChisq = cv::compareHist(hist1, hist1, CV_COMP_CHISQR);
//     float compIntersect = cv::compareHist(hist1, hist1, CV_COMP_INTERSECT);
    float compBhatta = cv::compareHist(hist1, hist2, CV_COMP_BHATTACHARYYA);
//     float compHellinger = cv::compareHist(hist1, hist2, CV_COMP_HELLINGER);
//     
//     cout << "compCorrel " << compCorrel << ", compChisq " << compChisq << ", compIntersect " << compIntersect 
//     << ", compBhatta " << compBhatta << ", compHellinger " << compHellinger << endl;
    
    
//     return (1.0 - compCorrel);
    return compBhatta;
}

float StixelsTracker::compareHistograms(const cv::Mat& img1, const cv::Mat& img2, const cv::Rect& rect1, const cv::Rect& rect2)
{
    cv::Mat roiImg1 = img1(rect1);
    cv::Mat roiImg2 = img2(rect2);
    
//     cv::imshow("roiImg1", roiImg1);
//     cv::imshow("roiImg2", roiImg2);
    
    const int histSize = 255;
    
    cv::cvtColor(roiImg1, roiImg1, CV_BGR2GRAY);
    cv::cvtColor(roiImg2, roiImg2, CV_BGR2GRAY);
    
    cv::Mat hist1, hist2;
    cv::calcHist(&roiImg1, 1, 0, cv::Mat(), hist1, 1, &histSize, 0);
    cv::calcHist(&roiImg2, 1, 0, cv::Mat(), hist2, 1, &histSize, 0);
    normalize(hist1, hist1, 0, 255, CV_MINMAX, CV_32F);
    normalize(hist2, hist2, 0, 255, CV_MINMAX, CV_32F);
    
//     cout << "hist " << cv::compareHist(hist1, hist2, CV_COMP_BHATTACHARYYA) << endl;
    
//     cv::waitKey(0);
    
    return cv::compareHist(hist1, hist2, CV_COMP_BHATTACHARYYA);
}

void StixelsTracker::computeObstacles() {
    vector <int> discontPrev, discontCurr;
    
    for (uint32_t i = 1; i < current_stixels_p->size(); i++) {
        if (previous_stixels_p->at(i).bottom_y != previous_stixels_p->at(i - 1).bottom_y) {
            discontPrev.push_back(i);
        }
        if (current_stixels_p->at(i).bottom_y != current_stixels_p->at(i - 1).bottom_y) {
            discontCurr.push_back(i);
        }
    }
    
    m_prevObstacleCorresp.swap(m_currObstacleCorresp);
    m_currObstacleCorresp = vector<int>(current_stixels_p->size(), -1);
    
    cv::Mat diffPolarMappedGray, polarPrevGray, polarCurrGray;
    if (mp_polarCalibration) {
        cv::Mat polarOutput;
        cv::Mat polar1, polar2, polarPrev, polarCurr, diffPolar, diffPolarMapped;
        mp_polarCalibration->getStoredRectifiedImages(polar1, polar2);
        cv::Mat inverseX, inverseY;
        mp_polarCalibration->getInverseMaps(inverseX, inverseY, 1);
        cv::remap(polar1, polarPrev, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        cv::remap(polar2, polarCurr, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        
//         cv::subtract(polar1, polar2, diffPolar);
//         cv::remap(diffPolar, diffPolarMapped, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        
//         cv::cvtColor(diffPolarMapped, diffPolarMappedGray, CV_BGR2GRAY);
        cv::cvtColor(polarPrev, polarPrevGray, CV_BGR2GRAY);
        cv::cvtColor(polarCurr, polarCurrGray, CV_BGR2GRAY);
        
        polarPrevGray.convertTo(polarPrevGray, CV_64F);
        polarCurrGray.convertTo(polarCurrGray, CV_64F);
    }
    
    m_obstacles.clear();
    int & stixelIdxL = discontCurr[0];
    Stixel stixelL = current_stixels_p->at(stixelIdxL);
    Stixel3d stixel3dL(stixelL);
    stixel3dL.update3dcoords(stereo_camera);
    t_obstacle currObstacle;
    currObstacle.stixels.push_back(stixel3dL);
    for (uint32_t i = 0; i < discontCurr.size(); i++) {
        int & stixelIdxR = discontCurr[i];
        
        const Stixel stixelR = current_stixels_p->at(stixelIdxR);
        Stixel3d stixel3dR(stixelR);
        stixel3dR.update3dcoords(stereo_camera);
        
        
        if (fabs(stixel3dR.bottom3d.z - stixel3dL.bottom3d.z) > 1.0) {
            // TODO: Get values from stored stixels
            currObstacle.roi.x = stixelL.x;
            currObstacle.roi.y = stixelL.top_y;
            currObstacle.roi.height = stixelL.bottom_y;
            currObstacle.roi3d.mean = cv::Point3d(0, 0, 0);
            currObstacle.roi3d.min = cv::Point3d(numeric_limits<double>::max(), 0.0, numeric_limits<double>::max());
            currObstacle.roi3d.max = -cv::Point3d(numeric_limits<double>::max(), numeric_limits<double>::max(), numeric_limits<double>::max());
            vector<int> disparities(128, 0);
            vector <double> depths;
            for (uint32_t j = stixel3dL.getBottom2d<cv::Point2d>().x; j < stixel3dR.getBottom2d<cv::Point2d>().x; j++) {
                const Stixel stixel = current_stixels_p->at(j);
                Stixel3d stixel3d(stixel);
                stixel3d.update3dcoords(stereo_camera);
                currObstacle.stixels.push_back(stixel3d);
                
                currObstacle.roi.y = min(stixel.top_y, currObstacle.roi.y);
                currObstacle.roi.height = max(currObstacle.roi.height, stixel.bottom_y);
                
                currObstacle.roi3d.mean += stixel3d.bottom3d;
                
                currObstacle.roi3d.min.x = min(currObstacle.roi3d.min.x, stixel3d.bottom3d.x);
                currObstacle.roi3d.min.z = min(currObstacle.roi3d.min.z, stixel3d.bottom3d.z);

                currObstacle.roi3d.max.x = max(currObstacle.roi3d.max.x, stixel3d.bottom3d.x);
                currObstacle.roi3d.max.y = max(currObstacle.roi3d.max.y, (stixel3d.bottom3d.y - stixel3d.top3d.y));
                currObstacle.roi3d.max.z = max(currObstacle.roi3d.max.z, stixel3d.bottom3d.z);
                
                if (stixel.disparity < disparities.size())
                    disparities[stixel.disparity]++;
                
                depths.push_back(stixel3d.bottom3d.z);
            }
            currObstacle.roi.width = stixelR.x - stixelL.x;
            currObstacle.roi.height -= currObstacle.roi.y - 1;
            currObstacle.roi3d.mean.x /= currObstacle.stixels.size();
            currObstacle.roi3d.mean.y = 0.0;
            currObstacle.roi3d.mean.z /= currObstacle.stixels.size();
            
            currObstacle.roi3d.width = currObstacle.roi3d.max.x - currObstacle.roi3d.min.x;
            currObstacle.roi3d.height = currObstacle.roi3d.max.y;
            currObstacle.roi3d.length = currObstacle.roi3d.max.z - currObstacle.roi3d.min.z;

            currObstacle.roi3d.centroid.x = (currObstacle.roi3d.max.x + currObstacle.roi3d.min.x) / 2.0;
            currObstacle.roi3d.centroid.y = 0.0;
            currObstacle.roi3d.centroid.z = (currObstacle.roi3d.max.z + currObstacle.roi3d.min.z) / 2.0;
            
            currObstacle.roi3d.stdDev = 0.0;
            BOOST_FOREACH(const double & depth, depths) {
                currObstacle.roi3d.stdDev += (depth - currObstacle.roi3d.mean.z) * (depth - currObstacle.roi3d.mean.z);
            }
            currObstacle.roi3d.stdDev = sqrt(currObstacle.roi3d.stdDev / depths.size());
            
            currObstacle.disparity = *max_element(disparities.begin(),disparities.end());
            
            // TODO: Parameterize
            // TODO: Use 3d width and height to filter
            if ((currObstacle.roi.width > 10) &&
                (currObstacle.roi.height > 40)) {
                
                BOOST_FOREACH(const Stixel3d & stixel, currObstacle.stixels) {
                    m_currObstacleCorresp[stixel.getBottom2d<cv::Point2i>().x] = m_obstacles.size();
                }
                m_obstacles.push_back(currObstacle); 
            }
            
            stixelL = stixelR;
            stixel3dL = stixel3dR;
            currObstacle.stixels.clear();
            currObstacle.stixels.push_back(stixel3dL);
        }
    }
}

double StixelsTracker::getNcc(const cv::Mat& img1, const cv::Mat& img2, 
                              const cv::Rect& rect1, const cv::Rect& rect2)
{
    // Get the polar SAD for the obstacle
    cv::Mat roiPrev = img1(rect1);
    cv::Mat roiCurr = img2(rect2);
    
    cv::Size newSize(min(rect1.width, rect2.width), min(rect1.height, rect2.height));
    cv::resize(roiPrev, roiPrev, newSize, 0, 0, cv::INTER_CUBIC);
    cv::resize(roiCurr, roiCurr, newSize, 0, 0, cv::INTER_CUBIC);
    
    cv::cvtColor(roiPrev, roiPrev, CV_BGR2GRAY);
    cv::cvtColor(roiCurr, roiCurr, CV_BGR2GRAY);
    roiPrev.convertTo(roiPrev, CV_64F);
    roiCurr.convertTo(roiCurr, CV_64F);

    cv::Scalar meanCurr, meanPrev;
    cv::Scalar stddevCurr, stddevPrev;
    
    cv::meanStdDev (roiPrev, meanPrev, stddevPrev);
    cv::meanStdDev (roiCurr, meanCurr, stddevCurr);
    
    cv::Mat elemPrev = roiPrev - meanPrev;
    cv::Mat elemCurr = roiCurr - meanCurr;
    
    cv::Mat multiplication;
    
    cv::multiply(elemPrev, elemCurr, multiplication);
    cv::divide(multiplication, stddevPrev * stddevCurr, multiplication);
    
    return cv::mean(multiplication)[0];
}

void StixelsTracker::trackObstacles()
{
    vector < t_obstacle> prevObstacles;
    
    m_obstacles.swap(prevObstacles);

    computeObstacles();
    
    if (prevObstacles.size() == 0) {
        m_obstaclesTracker.resize(m_obstacles.size());
        
        for (uint32_t i = 0; i < m_obstacles.size(); i++) {
            m_obstaclesTracker[i].push_front(m_obstacles[i]);
        }
        
        return;
    }
    
    cv::Mat lastImg, currImg;
    gil2opencv(previous_image_view, lastImg);
    gil2opencv(current_image_view, currImg);

    cv::Mat polarPrevGray, polarCurrGray;
    if (mp_polarCalibration) {
        cv::Mat polarOutput;
        cv::Mat polar1, polar2, polarPrev, polarCurr, diffPolar, diffPolarMapped;
        mp_polarCalibration->getStoredRectifiedImages(polar1, polar2);
        cv::Mat inverseX, inverseY;
        mp_polarCalibration->getInverseMaps(inverseX, inverseY, 1);
        cv::remap(polar1, polarPrev, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        cv::remap(polar2, polarCurr, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        
        cv::cvtColor(polarPrev, polarPrevGray, CV_BGR2GRAY);
        cv::cvtColor(polarCurr, polarCurrGray, CV_BGR2GRAY);
        
        polarPrevGray.convertTo(polarPrevGray, CV_64F);
        polarCurrGray.convertTo(polarCurrGray, CV_64F);
    }
    
    lemon::SmartGraph graph;
    lemon::SmartGraph::EdgeMap <double> costs(graph);
    lemon::SmartGraph::NodeMap <uint32_t> nodeIdx(graph);
    graph.reserveNode(prevObstacles.size() + m_obstacles.size());
    graph.reserveEdge(prevObstacles.size() * m_obstacles.size());
    
    for (uint32_t i = 0; i < prevObstacles.size() + m_obstacles.size(); i++) {
        nodeIdx[graph.addNode()] = i;
    }

    cv::Mat correspondences = cv::Mat::ones(prevObstacles.size(), m_obstacles.size(), CV_64FC1) * -1;
    for (uint32_t i = 0; i < m_obstacles.size(); i++) {
        for (uint32_t j = 0; j < prevObstacles.size(); j++) {
            if (cv::norm(m_obstacles[i].roi3d.centroid - prevObstacles[j].roi3d.centroid) < 1.0)
                correspondences.at<double>(j, i) = 1 - compareHistograms(lastImg, currImg, prevObstacles[j].roi, m_obstacles[i].roi);
        }        
    }

    // Gets the number of correspondences between obstacles
//     cv::Mat correspondences = cv::Mat::ones(prevObstacles.size(), m_obstacles.size(), CV_32SC1) * -1;
//     for (uint32_t i = 0; i < m_obstacles.size(); i++) {
//         BOOST_FOREACH(const Stixel3d & currStixel, m_obstacles[i].stixels) {
//             const cv::Point2i & currPoint = currStixel.getBottom2d<cv::Point2i>();
//             if (stixels_motion[currPoint.x] != -1) {
//                 const Stixel & prevStixel = previous_stixels_p->at(stixels_motion[currPoint.x]);
//                 
//                 if (m_prevObstacleCorresp[prevStixel.x] != -1) {
//                     correspondences.at<int>(m_prevObstacleCorresp[prevStixel.x], i) = 
//                         correspondences.at<int>(m_prevObstacleCorresp[prevStixel.x], i) + 1;
//                         
//                 }
//             }
//         }
//     }
    
//     cout << "   ";
//     for (uint32_t j = 0; j < m_obstacles.size(); j++) {
//         cout << j << "   ";
//     }
//     cout << endl;
//     for (uint32_t i = 0; i < prevObstacles.size(); i++) {
//         cout << i << ": ";
//         for (uint32_t j = 0; j < m_obstacles.size(); j++) {
//             cout << correspondences.at<double>(i, j) << "   ";
//         }
//         cout << endl;
//     }
//     cout << endl;
    
    for (uint32_t i = 0; i < prevObstacles.size(); i++) {
        for (uint32_t j = 0; j < m_obstacles.size(); j++) {
//             const lemon::SmartGraph::Edge e = graph.addEdge(graph.addNode(), graph.addNode());
            const lemon::SmartGraph::Edge e = graph.addEdge(graph.nodeFromId(i), graph.nodeFromId((uint32_t)(j + prevObstacles.size())));
            costs[e] = correspondences.at<double>(i, j);
        }
    }
    
    lemon::MaxWeightedMatching< lemon::SmartGraph, lemon::SmartGraph::EdgeMap <double> > graphMatcher(graph, costs);
    
    graphMatcher.run();
    
    const lemon::SmartGraph::NodeMap<lemon::SmartGraph::Arc> & matchingMap = graphMatcher.matchingMap();
    
    vector < deque < t_obstacle> > prevObstaclesTracker;
    prevObstaclesTracker.swap(m_obstaclesTracker);
    
    m_obstaclesTracker.clear();
    m_obstaclesTracker.resize(m_obstacles.size());
    
    for (uint32_t i = 0; i < prevObstacles.size(); i++) {
        if (graphMatcher.mate(graph.nodeFromId(i)) != lemon::INVALID) {
            lemon::SmartGraph::Arc arc = matchingMap[graph.nodeFromId(i)];
//             cout << i << " -> " << graph.id(graph.target(arc)) - prevObstacles.size() << endl;
            int currIdx = graph.id(graph.target(arc)) - prevObstacles.size();
            deque <t_obstacle> & track = prevObstaclesTracker[i];
            track.push_front(m_obstacles[currIdx]);
            
            m_obstaclesTracker[currIdx] = track;
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // VISUALIZATION
    ////////////////////////////////////////////////////////////////////////////////////////////////////
//     return;
    cv::Mat img = cv::Mat::zeros(m_currImg.rows * 2, m_currImg.cols * 2, CV_8UC3);
    cv::Rect roi(0, 0, m_currImg.cols, m_currImg.rows);
    cv::Mat roiImgPrev = img(roi);
//     cv::Mat lastImg;
    gil2opencv(previous_image_view, lastImg);
    lastImg.copyTo(roiImgPrev);
    roi = cv::Rect(m_currImg.cols, 0, m_currImg.cols, m_currImg.rows);
    cv::Mat roiImgCurr = img(roi);
    m_currImg.copyTo(roiImgCurr);
    roi = cv::Rect(0, m_currImg.rows, m_currImg.cols, m_currImg.rows);
    cv::Mat roiImgTrack = img(roi);
    m_currImg.copyTo(roiImgTrack);
    roi = cv::Rect(m_currImg.cols, m_currImg.rows, m_currImg.cols, m_currImg.rows);
    cv::Mat roiImgPolar = img(roi);
//     cv::Mat roiImgAggr = img(roi);
//     m_currImg.copyTo(roiImgAggr);
    
    for (uint32_t i = 0; i < m_obstacles.size(); i++) {
        cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        cv::rectangle(roiImgCurr, cv::Point2i(m_obstacles[i].roi.x, m_obstacles[i].roi.y),
                      cv::Point2i(m_obstacles[i].roi.x + m_obstacles[i].roi.width, m_obstacles[i].roi.y + m_obstacles[i].roi.height),
                      color, 1);
        
        cv::rectangle(roiImgCurr, cv::Point2i(m_obstacles[i].roi.x, m_obstacles[i].roi.y),
                      cv::Point2i(m_obstacles[i].roi.x + 40, m_obstacles[i].roi.y - 10), cv::Scalar::all(255), -1);
        stringstream oss;
        oss << i;
        cv::putText(roiImgCurr, oss.str(), cv::Point2i(m_obstacles[i].roi.x, m_obstacles[i].roi.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(0));

        BOOST_FOREACH(const Stixel3d & currStixel, m_obstacles[i].stixels) {
            const cv::Point2i & currPoint = currStixel.getBottom2d<cv::Point2i>();
            if (stixels_motion[currPoint.x] != -1) {
                const Stixel & prevStixel = previous_stixels_p->at(stixels_motion[currPoint.x]);
                cv::line(roiImgCurr, currPoint, cv::Point2i(prevStixel.x, prevStixel.bottom_y), color, 1);
                cv::line(roiImgPrev, currPoint, cv::Point2i(prevStixel.x, prevStixel.bottom_y), color, 1);
            }
        }
    }
    for (uint32_t i = 0; i < prevObstacles.size(); i++) {
        cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        cv::rectangle(roiImgPrev, cv::Point2i(prevObstacles[i].roi.x, prevObstacles[i].roi.y),
                  cv::Point2i(prevObstacles[i].roi.x + prevObstacles[i].roi.width, prevObstacles[i].roi.y + prevObstacles[i].roi.height),
                  color, 1);
        cv::rectangle(roiImgPrev, cv::Point2i(prevObstacles[i].roi.x, prevObstacles[i].roi.y),
                      cv::Point2i(prevObstacles[i].roi.x + 20, prevObstacles[i].roi.y - 10), cv::Scalar::all(255), -1);
        stringstream oss;
        oss << i;
        cv::putText(roiImgPrev, oss.str(), cv::Point2i(prevObstacles[i].roi.x, prevObstacles[i].roi.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(0));
    }
    
    BOOST_FOREACH(const deque<t_obstacle> & track, m_obstaclesTracker) {
        cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        
        double speed = 0.0;
        for (uint32_t i = 1; i < track.size(); i++) {
            cv::line(roiImgTrack, cv::Point2i(track[i].roi.x, track[i].roi.y), 
                     cv::Point2i(track[i - 1].roi.x, track[i - 1].roi.y), color, 1);
            cv::line(roiImgTrack, cv::Point2i(track[i].roi.x + track[i].roi.width, track[i].roi.y), 
                     cv::Point2i(track[i - 1].roi.x + track[i - 1].roi.width, track[i - 1].roi.y), color, 1);
            cv::line(roiImgTrack, cv::Point2i(track[i].roi.x, track[i].roi.y + track[i].roi.height), 
                     cv::Point2i(track[i - 1].roi.x, track[i - 1].roi.y + track[i - 1].roi.height), color, 1);
            cv::line(roiImgTrack, cv::Point2i(track[i].roi.x + track[i].roi.width, track[i].roi.y + track[i].roi.height), 
                     cv::Point2i(track[i - 1].roi.x + track[i - 1].roi.width, track[i - 1].roi.y + track[i - 1].roi.height), color, 1);
            cv::rectangle(roiImgTrack, cv::Point2i(track[i].roi.x, track[i].roi.y),
                          cv::Point2i(track[i].roi.x + track[i].roi.width, track[i].roi.y + track[i].roi.height),
                          color, 1);
            speed += sqrt((track[i].roi3d.mean.x - track[i - 1].roi3d.mean.x) * (track[i].roi3d.mean.x - track[i - 1].roi3d.mean.x) +
//                           (track[i].roi3d.mean.y - track[i - 1].roi3d.mean.y) * (track[i].roi3d.mean.y - track[i - 1].roi3d.mean.y) + 
                          (track[i].roi3d.mean.z - track[i - 1].roi3d.mean.z) * (track[i].roi3d.mean.z - track[i - 1].roi3d.mean.z));
        }
        speed /= track.size() * 0.07692307692307692308;
        if (track.size() != 0) {
            cv::rectangle(roiImgTrack, cv::Point2i(track[0].roi.x, track[0].roi.y),
                      cv::Point2i(track[0].roi.x + track[0].roi.width, track[0].roi.y + track[0].roi.height),
                      color, 1);
            
            t_obstacle lastObstacle = track[track.size() - 1];
            cv::rectangle(roiImgTrack, cv::Point2i(lastObstacle.roi.x, lastObstacle.roi.y),
                          cv::Point2i(lastObstacle.roi.x + 20, lastObstacle.roi.y - 10), cv::Scalar::all(255), -1);
            stringstream oss;
            oss << speed;
            cv::putText(roiImgTrack, oss.str(), cv::Point2i(lastObstacle.roi.x, lastObstacle.roi.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(0));
        }
    }
    
    // Polar calibration visualization
    if (mp_polarCalibration) {
        cv::Mat diffPolarGray, polarPrevGray, polarCurrGray;
        cv::Mat polarOutput;
        cv::Mat polar1, polar2, polarPrev, polarCurr, diffPolar, diffPolarMapped;
        mp_polarCalibration->getStoredRectifiedImages(polar1, polar2);
        cv::Mat inverseX, inverseY;
        mp_polarCalibration->getInverseMaps(inverseX, inverseY, 1);
        cv::remap(polar1, polarPrev, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        cv::remap(polar2, polarCurr, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        
        cv::absdiff(polar1, polar2, diffPolar);
        cv::Mat mask1(polar1.rows, polar1.cols, CV_8UC1);
        cv::Mat mask2(polar2.rows, polar2.cols, CV_8UC1);
        mask1.setTo(cv::Scalar(255));
        mask2.setTo(cv::Scalar(255));
        
        cv::remap(diffPolar, diffPolar, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        cv::remap(polar1, polar1, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        cv::remap(polar2, polar2, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        cv::remap(mask1, mask1, inverseX, inverseY, cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
        cv::remap(mask2, mask2, inverseX, inverseY, cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
        
        cv::Mat mask;
        cv::bitwise_and(mask1, mask2, mask);
        
        cv::threshold(mask, mask, 254, 255, cv::THRESH_BINARY);
        
        cv::Mat maskCopy;
        cv::normalize(mask, maskCopy, 0, 1, CV_MINMAX, CV_8U);
        cv::cvtColor(maskCopy, maskCopy, CV_GRAY2BGR);
        
        cv::multiply(diffPolar, maskCopy, diffPolar);
        cv::multiply(polar1, maskCopy, polar1);
        cv::multiply(polar2, maskCopy, polar2);
        diffPolar.copyTo(diffPolar, mask);
        
        cv::cvtColor(diffPolar, diffPolarGray, CV_BGR2GRAY);
        cv::cvtColor(polarPrev, polarPrevGray, CV_BGR2GRAY);
        cv::cvtColor(polarCurr, polarCurrGray, CV_BGR2GRAY);
        
        cv::threshold(diffPolarGray, diffPolarGray, 50, 255, cv::THRESH_BINARY);
        
        cv::cvtColor(diffPolarGray, diffPolar, CV_GRAY2BGR);
        
        diffPolar.copyTo(roiImgPolar);
//         polar1.copyTo(roiImgPolar);
//         m_currImg.copyTo(roiImgPolar);
        
        // TODO: Parameterize
        const double gridSize = 0.10;
        
//         lastImg.copyTo(roiImgTrack);
        for (uint32_t i = 0; i < m_obstacles.size(); i++) {
            cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
            double ncc = getNcc(polar1, polar2, m_obstacles[i].roi, m_obstacles[i].roi);
            double absdiff = cv::mean(diffPolarGray(m_obstacles[i].roi))[0];
            double bat = compareHistograms(polar1, polar2, m_obstacles[i].roi, m_obstacles[i].roi);
            
            cv::Mat obstacle = diffPolarGray(m_obstacles[i].roi);
            
            cv::Mat occupancyMap = cv::Mat::zeros(ceil(m_obstacles[i].roi3d.height / gridSize), ceil(m_obstacles[i].roi3d.width / gridSize), CV_8UC1);
            
            double factorX = (double)occupancyMap.cols / obstacle.cols;
            double factorY = (double)occupancyMap.rows / obstacle.rows;
            for (uint32_t y = obstacle.rows / 2.0; y < obstacle.rows; y++)  {
                for (uint32_t x = 0; x < obstacle.cols; x++)  {
                    if (obstacle.at<uchar>(y, x) != 0) {
                        occupancyMap.at<uchar>(y * factorY, x * factorX) = 0xFF;
                    }
                }
            }
            
            occupancyMap = occupancyMap(cv::Rect(0, occupancyMap.rows / 2.0, occupancyMap.cols, ceil(occupancyMap.rows / 2.0)));
            
//             cout << i << ": " << cv::mean(occupancyMap)[0] << ", " << cv::sum(occupancyMap)[0] << endl;
            
//             if (cv::countNonZero(diffPolarGray(m_obstacles[i].roi)) > 10)
            if (cv::mean(occupancyMap)[0] > 100)
                cv::rectangle(roiImgPolar, cv::Point2i(m_obstacles[i].roi.x, m_obstacles[i].roi.y),
                          cv::Point2i(m_obstacles[i].roi.x + m_obstacles[i].roi.width, m_obstacles[i].roi.y + m_obstacles[i].roi.height),
                          cv::Scalar(0, 255, 0), 2);
            else
                cv::rectangle(roiImgPolar, cv::Point2i(m_obstacles[i].roi.x, m_obstacles[i].roi.y),
                              cv::Point2i(m_obstacles[i].roi.x + m_obstacles[i].roi.width, m_obstacles[i].roi.y + m_obstacles[i].roi.height),
                              cv::Scalar(0, 0, 255), 2);
            
//             cout << i << ": ncc " << ncc << ", absdiff " << absdiff << ", bat " << bat << endl;
//                 transformPoints(const vector< cv::Point2d >& points1, 
//                                 vector< cv::Point2d >& transformedPoints1, const uint8_t & whichImage);
//                 vector< cv::Point2d > points(4);
//                 points[0] = cv::Point2d(m_obstacles[i].roi.x, m_obstacles[i].roi.y);
//                 points[1] = cv::Point2d(m_obstacles[i].roi.x + m_obstacles[i].roi.width, m_obstacles[i].roi.y);
//                 points[2] = cv::Point2d(m_obstacles[i].roi.x, m_obstacles[i].roi.y + m_obstacles[i].roi.height);
//                 points[3] = cv::Point2d(m_obstacles[i].roi.x + m_obstacles[i].roi.width, m_obstacles[i].roi.y + m_obstacles[i].roi.height);
//                 
//                 vector< cv::Point2d > transformedPoints;
//                 vector< cv::Point2d > transformedPoints2;
//             
//                 mp_polarCalibration->transformPoints(points, transformedPoints, 1);
//                 mp_polarCalibration->transformPoints(points, transformedPoints2, 2);
// 
//                 cv::rectangle(roiImgPolar, points[0], points[3], cv::Scalar(255, 0, 0), 1);
//                 
//                 cv::Point2d centroidCurr, centroidPrev;
//                 double polarDist = 0.0;
//                 for (uint32_t j = 0; j < 4; j++) {
//                     centroidCurr += points[j];
//                     centroidPrev += transformedPoints[j];
//                     polarDist += fabs(transformedPoints[j].x - transformedPoints2[j].x);
//                     cout << transformedPoints2[j] << " -- " << transformedPoints[j] << endl;
//                 }
//                 
//                 polarDist /= 4.0;
//                 
//                 cv::rectangle(roiImgTrack, transformedPoints[0], transformedPoints[3], cv::Scalar(255, 0, 0), 1);
//                 
//                 centroidCurr.x /= 4.0;
//                 centroidCurr.y /= 4.0;
//                 centroidPrev.x /= 4.0;
//                 centroidPrev.y /= 4.0;
//                 
//                 Eigen::Vector2f tmpCentroidCurr2d, tmpCentroidPrev2d;
//                 tmpCentroidCurr2d << centroidCurr.x, centroidCurr.y;
//                 tmpCentroidPrev2d << centroidPrev.x, centroidPrev.y;
//                 
//                 const float & depth = stereo_camera.disparity_to_depth(m_obstacles[i].disparity);
//                 const Eigen::Vector3f & tmpCentroidCurr3d = stereo_camera.get_left_camera().back_project_2d_point_to_3d(tmpCentroidCurr2d, depth);
//                 const Eigen::Vector3f & tmpCentroidPrev3d = stereo_camera.get_left_camera().back_project_2d_point_to_3d(tmpCentroidPrev2d, depth);
//                 
//                 cv::Point3d centroidCurr3d = cv::Point3d(tmpCentroidCurr3d[0], tmpCentroidCurr3d[1], tmpCentroidCurr3d[2]);
//                 cv::Point3d centroidPrev3d = cv::Point3d(tmpCentroidPrev3d[0], tmpCentroidPrev3d[1], tmpCentroidPrev3d[2]);
//                 
//                 const double & dist = cv::norm(centroidCurr3d - centroidPrev3d);
//                 
//                 cout << i << ": " << centroidCurr << ", " << centroidPrev << " --> " << centroidCurr3d << ", " << centroidPrev3d << " = " << dist << ", " << polarDist << endl;
        }
    }

    // Aggregation
//     cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
//     uint32_t count = 1;
//     for (uint32_t i = 1; i < m_obstacles.size(); i++) {
//         if ((fabs(m_obstacles[i].roi3d.min.z - m_obstacles[i - 1].roi3d.min.z) > 1.0) ||
//             ((m_obstacles[i].roi3d.min.x - m_obstacles[i - 1].roi3d.max.x) > 0.20)) {
//             count++;
//             color = cv::Scalar(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
//         }
//         
//         cout << i << ": " << m_obstacles[i].roi3d.min << ", " << 
//                             m_obstacles[i].roi3d.max << ", " << 
//                             m_obstacles[i].roi3d.mean << endl;
//         cout << i << ": " << m_obstacles[i].roi3d.width << ", " <<
//                             m_obstacles[i].roi3d.height << ", " <<
//                             m_obstacles[i].roi3d.length << "; " <<
//                             m_obstacles[i].roi3d.stdDev << endl;
//         if (((m_obstacles[i].roi3d.width) > 0.5) &&
//             ((m_obstacles[i].roi3d.height) > 1.5)) {
//             
//             cv::rectangle(roiImgAggr, cv::Point2i(m_obstacles[i].roi.x, m_obstacles[i].roi.y),
//                             cv::Point2i(m_obstacles[i].roi.x + m_obstacles[i].roi.width, m_obstacles[i].roi.y + m_obstacles[i].roi.height),
//                             color, 1);
//         }
//     }
//     cout << "Total " << count << endl;

    cv::imshow("StixelsEvol", img);
}
