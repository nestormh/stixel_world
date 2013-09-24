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


#include "extendedfastgroundplaneestimator.h"

#include "extendedfastgroundplaneestimator.h"

#include <boost/foreach.hpp>
#include <iostream>

namespace stixel_world {

typedef doppia::FastGroundPlaneEstimator::points_t points_t;
    
using namespace std;
    
ExtendedFastGroundPlaneEstimator::ExtendedFastGroundPlaneEstimator(
    const boost::program_options::variables_map &options,
    const doppia::StereoCameraCalibration &stereo_calibration) :
        FastGroundPlaneEstimator(options, stereo_calibration)
{

    
    
}

ExtendedFastGroundPlaneEstimator::~ExtendedFastGroundPlaneEstimator()
{

}

void set_points_weights(const points_t &points,
                        const Eigen::VectorXf &row_weights,
                        Eigen::VectorXf &points_weights)
{
    points_weights.setOnes(points.size());
    
    int i=0;
    BOOST_FOREACH(points_t::const_reference point, points)
    {
        const int &point_y = point.second;
        points_weights(i) = row_weights(point_y);
        i+=1;
    }
    
    return;
}

void ExtendedFastGroundPlaneEstimator::compute()
{    
    static int num_iterations = 0;
    static double cumulated_time = 0;
    
    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();
    
    // compute v_disparity --
    compute_v_disparity_data();
    
    set_points_weights(points, row_weights, points_weights);
    // compute line --
    estimate_ground_plane();
    
    confidence_is_up_to_date = false;
    
    // timing ---
    cumulated_time += omp_get_wtime() - start_wall_time;
    num_iterations += 1;
    
    const doppia::GroundPlane & gp = get_ground_plane();
    cout << "H = " << gp.get_height() << ", P = " << gp.get_pitch() << endl;
//     cout << "Ground plane " << get_ground_plane() << endl;
    
    if((silent_mode == false) and ((num_iterations % num_iterations_for_timing) == 0))
    {
        printf("Average FastGroundPlaneEstimator::compute speed  %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations / cumulated_time, num_iterations );
    }
    
//     doppia::GroundPlane ground_plane_prior;
//     ground_plane_prior.set_from_metric_units(
//         -0.05f, 0.0, 0.98);
//     estimated_ground_plane = ground_plane_prior;
    
//     const doppia::GroundPlane & gp3 = get_ground_plane();
//     cout << "H = " << gp3.get_height() << ", P = " << gp3.get_pitch() << endl;
    
    return;
}

}
