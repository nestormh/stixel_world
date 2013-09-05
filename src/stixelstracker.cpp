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
    compute_maximum_pixelwise_motion_for_stixel_lut();
    
    m_sad_factor = 0.3;
    m_height_factor = 0.0;
    m_polar_dist_factor = 0.7;
}

void StixelsTracker::set_motion_cost_factors(const float& sad_factor, const float& height_factor, const float& polar_dist_factor)
{
    if ((m_sad_factor + m_height_factor + m_polar_dist_factor) == 1.0) {
        m_sad_factor = sad_factor;
        m_height_factor = height_factor;
        m_polar_dist_factor = polar_dist_factor;
    } else {
        cerr << "The sum of motion cost factors should be 1!!!" << endl;
    }
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

cv::Point2d StixelsTracker::get_polar_point(const cv::Mat& mapX, const cv::Mat& mapY, const Stixel stixel)
{
    return cv::Point2d(mapX.at<float>(stixel.bottom_y, stixel.x),
                       mapY.at<float>(stixel.bottom_y, stixel.x));
}


void StixelsTracker::compute()
{
    compute_motion_cost_matrix();
    compute_motion();
    update_stixel_tracks_image();
    updateTracker();
    
    return;
}

void StixelsTracker::compute_motion_cost_matrix()
{    
    
    const double & startWallTime = omp_get_wtime();
    
    const float maximum_depth_difference = 1.0;
    
    const float maximum_allowed_real_height_difference = 0.5f;
    const float maximum_allowed_polar_distance = 50.0f;
    
    assert((m_sad_factor + m_height_factor + m_polar_dist_factor) == 1.0);
    
    const float maximum_real_motion = maximum_pedestrian_speed / video_frame_rate;
    
    const unsigned int number_of_current_stixels = current_stixels_p->size();
    const unsigned int number_of_previous_stixels = previous_stixels_p->size();
    
    cv::Mat mapXprev, mapYprev, mapXcurr, mapYcurr;
    mp_polarCalibration->getInverseMaps(mapXprev, mapYprev, 1);
    mp_polarCalibration->getInverseMaps(mapXcurr, mapYcurr, 2);
    
    motion_cost_matrix.fill( 0.f );
    pixelwise_sad_matrix.fill( 0.f );
    real_height_differences_matrix.fill( 0.f );
    m_stixelsPolarDistMatrix.fill(0.f);
    motion_cost_assignment_matrix.fill( false );
    
    current_stixel_depths.fill( 0.f );
    current_stixel_real_heights.fill( 0.f );
    
    
    // Fill in the motion cost matrix
    #pragma omp parallel for schedule(dynamic)
    for( unsigned int s_current = 0; s_current < number_of_current_stixels; ++s_current )
    {
        const Stixel& current_stixel = ( *current_stixels_p )[ s_current ];
        const cv::Point2d current_polar = get_polar_point(mapXcurr, mapYcurr, current_stixel);
        
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
                const cv::Point2d previous_polar = get_polar_point(mapXprev, mapYprev, previous_stixel);
                
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
                            
                            if( current_stixel.type != Stixel::Occluded && previous_stixel.type != Stixel::Occluded )
                            {
                                pixelwise_sad = compute_pixelwise_sad( current_stixel, previous_stixel, current_image_view, previous_image_view, stixel_horizontal_padding );
                                real_height_difference = fabs( current_stixel_real_height - compute_stixel_real_height( previous_stixel ) );
                                polar_distance = cv::norm(previous_polar - current_polar);
                            }
                            else
                            {
                                pixelwise_sad = maximum_pixel_value;
                                real_height_difference = maximum_allowed_real_height_difference;
                                polar_distance = maximum_allowed_polar_distance;
                            }
                            
                            pixelwise_sad_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = pixelwise_sad;
                            real_height_differences_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) =
                                    std::min( 1.0f, real_height_difference / maximum_allowed_real_height_difference );
                            
                            m_stixelsPolarDistMatrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = 
                                    std::min( 1.0f, polar_distance / maximum_allowed_polar_distance );
                            
                            motion_cost_assignment_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = true;
                        }
                    }
                }
                
            } // End of for( s_prev )
        }
        
    } // End of for( s_current )
    
    /// Rescale the real height difference matrix elemants so that it will have the same range with pixelwise_sad
    const float maximum_real_height_difference = real_height_differences_matrix.maxCoeff();
    //    real_height_differences_matrix = real_height_differences_matrix * ( float ( maximum_pixel_value ) / maximum_real_height_difference );
    real_height_differences_matrix = real_height_differences_matrix * maximum_pixel_value;
    
    m_stixelsPolarDistMatrix = m_stixelsPolarDistMatrix * maximum_pixel_value;
    
    /// Fill in the motion cost matrix
//     motion_cost_matrix = alpha * pixelwise_sad_matrix + ( 1 - alpha ) * real_height_differences_matrix; // [0, 255]
    motion_cost_matrix = m_sad_factor * pixelwise_sad_matrix + 
                         m_height_factor * real_height_differences_matrix +
                         m_polar_dist_factor * m_stixelsPolarDistMatrix;
                         
    const float maximum_cost_matrix_element = motion_cost_matrix.maxCoeff(); // Minimum is 0 by definition
    
    /// Fill in disappearing stixel entries specially
    //    insertion_cost_dp = maximum_cost_matrix_element * 0.75;
    insertion_cost_dp = maximum_pixel_value * 0.6;
    deletion_cost_dp = insertion_cost_dp; // insertion_cost_dp is not used for the moment !!
    
    {
        const unsigned int number_of_cols = motion_cost_matrix.cols();
        const unsigned int largest_row_index = motion_cost_matrix.rows() - 1;
        
    //     for( unsigned int j = 0, number_of_cols = motion_cost_matrix.cols(), largest_row_index = motion_cost_matrix.rows() - 1; j < number_of_cols; ++j )
//         #pragma omp parallel for schedule(dynamic)
        for( unsigned int j = 0; j < number_of_cols; ++j )
        {
            motion_cost_matrix( largest_row_index, j ) = deletion_cost_dp;
            motion_cost_assignment_matrix( largest_row_index, j ) = true;
            
        } // End of for(j)
    }
    
    {
        const unsigned int number_of_rows = motion_cost_matrix.rows();
        const unsigned int number_of_cols = motion_cost_matrix.cols();
                
//         for( unsigned int i = 0, number_of_rows = motion_cost_matrix.rows(); i < number_of_rows; ++i )
//         #pragma omp parallel for schedule(dynamic)
        for( unsigned int i = 0; i < number_of_rows; ++i )
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

void StixelsTracker::updateTracker()
{
    const stixels_t * currStixels = current_stixels_p;
    stixels_motion_t corresp = stixels_motion;
    
    if (m_tracker.size() == 0) {
        m_tracker.resize(currStixels->size());
        for (uint32_t i = 0; i < currStixels->size(); i++) {
            m_tracker[i].push_back(currStixels->at(i));
        }
    }
    
    t_tracker tmpTracker(m_tracker.size());
    copy(m_tracker.begin(), m_tracker.end(), tmpTracker.begin());
    m_tracker.clear();
    m_tracker.resize(currStixels->size());
    
    for (uint32_t i = 0; i < currStixels->size(); i++) {
        if (corresp[i] >= 0.0f) {
            m_tracker[i] = tmpTracker[corresp[i]];
            m_tracker[i].push_back(currStixels->at(i));
        }
    }
}

void StixelsTracker::drawTracker(cv::Mat& img)
{
    gil2opencv(current_image_view, img);
    

    cv::rectangle(img, cv::Point2d(0, 0), cv::Point2d(img.cols - 1, 20), cv::Scalar::all(0), -1);
    stringstream oss;
    oss << "SAD factor = " << m_sad_factor << ", Height factor = " << m_height_factor << ", Polar distance factor  = " << m_polar_dist_factor;
    cv::putText(img, oss.str(), cv::Point2d(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(255));
    
    for (t_tracker::iterator it = m_tracker.begin(); it != m_tracker.end(); it++) {
        const cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        if (it->size() != 0) {
            const Stixel lastStixel = it->at(it->size() - 1);
            cv::circle(img, cv::Point2d(lastStixel.x, lastStixel.bottom_y), 1, color);
        
            cv::Point2d lastPoint(it->at(0).x, it->at(0).bottom_y);
            for (stixels_t::iterator it2 = it->begin() + 1; it2 != it->end(); it2++) {
                const cv::Point2d currPoint(it2->x, it2->bottom_y);
                
                cv::line(img, lastPoint, currPoint, color);
                lastPoint = currPoint;
            }
        }
    }
}
