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

using namespace std;

const float MIN_FLOAT_DISPARITY = 0.8f;

StixelsTracker::StixelsTracker::StixelsTracker(const boost::program_options::variables_map& options, 
                                               const MetricStereoCamera& camera, int stixels_width) :
                                               DummyStixelMotionEstimator(options, camera, stixels_width)
{ 
    m_stixelsPolarDistMatrix = Eigen::MatrixXf::Zero( motion_cost_matrix.rows(), motion_cost_matrix.cols() ); // Matrix is initialized with 0.
}


void StixelsTracker::compute()
{
    compute_motion_cost_matrix();
    compute_motion();
    update_stixel_tracks_image();
    
    return;
}

void StixelsTracker::compute_motion_cost_matrix()
{    
    const float maximum_depth_difference = 1.0;
    
    const float maximum_allowed_real_height_difference = 0.5f;
    const float alpha = 0.3;
    
    const float maximum_real_motion = maximum_pedestrian_speed / video_frame_rate;
    
    const unsigned int number_of_current_stixels = current_stixels_p->size();
    const unsigned int number_of_previous_stixels = previous_stixels_p->size();
    
    motion_cost_matrix.fill( 0.f );
    pixelwise_sad_matrix.fill( 0.f );
    real_height_differences_matrix.fill( 0.f );
    m_stixelsPolarDistMatrix.fill(0.f);
    motion_cost_assignment_matrix.fill( false );
    
    current_stixel_depths.fill( 0.f );
    current_stixel_real_heights.fill( 0.f );
    
    // Fill in the motion cost matrix
    for( unsigned int s_current = 0; s_current < number_of_current_stixels; ++s_current )
    {
        const Stixel& current_stixel = ( *current_stixels_p )[ s_current ];
        
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
                
                if( previous_stixel.x - ( previous_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 &&
                    previous_stixel.x + ( previous_stixel.width - 1 ) / 2 + stixel_horizontal_padding < previous_image_view.width() /*&&
                    previous_stixel.type != Stixel::Occluded*/ ) // Horizontal padding for previous stixel is suitable
                {
                    const float previous_stixel_disparity = std::max< float >( MIN_FLOAT_DISPARITY, previous_stixel.disparity );
                    const float previous_stixel_depth = stereo_camera.disparity_to_depth( previous_stixel_disparity );
                    
                    Eigen::Vector3f real_motion = compute_real_motion_between_stixels( current_stixel, previous_stixel, current_stixel_depth, previous_stixel_depth );                    
                    const float real_motion_magnitude = real_motion.norm();
                    
                    //                    if( fabs( current_stixel_depth - previous_stixel_depth ) < maximum_depth_difference )
                    {
                        const int pixelwise_motion = previous_stixel.x - current_stixel.x; // Motion can be positive or negative
                        
                        const unsigned int maximum_motion_in_pixels_for_current_stixel = compute_maximum_pixelwise_motion_for_stixel( current_stixel );
                        
                        if( pixelwise_motion >= -( int( maximum_motion_in_pixels_for_current_stixel ) ) &&
                            pixelwise_motion <= int( maximum_motion_in_pixels_for_current_stixel ) /*&&
                            real_motion_magnitude <= maximum_real_motion*/ )
                        {
                            float pixelwise_sad;
                            float real_height_difference;
                            
                            if( current_stixel.type != Stixel::Occluded && previous_stixel.type != Stixel::Occluded )
                            {
                                pixelwise_sad = compute_pixelwise_sad( current_stixel, previous_stixel, current_image_view, previous_image_view, stixel_horizontal_padding );
                                real_height_difference = fabs( current_stixel_real_height - compute_stixel_real_height( previous_stixel ) );
                            }
                            else
                            {
                                pixelwise_sad = maximum_pixel_value;
                                real_height_difference = maximum_allowed_real_height_difference;
                            }
                            
                            pixelwise_sad_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = pixelwise_sad;
                            real_height_differences_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) =
                            std::min( 1.0f, real_height_difference / maximum_allowed_real_height_difference );
                            
                            motion_cost_assignment_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = true;
                            real_motion_vectors_matrix[ s_current ][ pixelwise_motion + maximum_possible_motion_in_pixels ] = real_motion;
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
    
    /// Fill in the motion cost matrix
    motion_cost_matrix = alpha * pixelwise_sad_matrix + ( 1 - alpha ) * real_height_differences_matrix; // [0, 255]
    
    const float maximum_cost_matrix_element = motion_cost_matrix.maxCoeff(); // Minimum is 0 by definition
    
    /// Fill in disappearing stixel entries specially
    //    insertion_cost_dp = maximum_cost_matrix_element * 0.75;
    insertion_cost_dp = maximum_pixel_value * 0.6;
    deletion_cost_dp = insertion_cost_dp; // insertion_cost_dp is not used for the moment !!
    
    for( unsigned int j = 0, number_of_cols = motion_cost_matrix.cols(), largest_row_index = motion_cost_matrix.rows() - 1; j < number_of_cols; ++j )
    {
        motion_cost_matrix( largest_row_index, j ) = deletion_cost_dp;
        motion_cost_assignment_matrix( largest_row_index, j ) = true;
        
    } // End of for(j)
    
    for( unsigned int i = 0, number_of_rows = motion_cost_matrix.rows(); i < number_of_rows; ++i )
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
    
    /**
     * 
     * Lines below are intended for DEBUG & VISUALIZATION purposes
     *
     **/
    
    //    fill_in_visualization_motion_cost_matrix();
    
    return;
}