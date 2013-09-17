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


#ifndef EXTENDEDSTIXELWORLDESTIMATORFACTORY_H
#define EXTENDEDSTIXELWORLDESTIMATORFACTORY_H

#include "stereo_matching/stixels/StixelWorldEstimatorFactory.hpp"

// using namespace doppia;

namespace stixel_world {
class ExtendedStixelWorldEstimatorFactory : public doppia::StixelWorldEstimatorFactory
{

    
public:
    static boost::program_options::options_description get_args_options();
    
    /// commonly used new_instance method
    static doppia::AbstractStixelWorldEstimator* new_instance(const boost::program_options::variables_map &options,
                                                      doppia::AbstractVideoInput &video_input);
    
    
    /// variant method used inside objects_detection_lib
    static doppia::AbstractStixelWorldEstimator* new_instance(const boost::program_options::variables_map &options,
                                                      const doppia::AbstractVideoInput::dimensions_t &input_dimensions,
                                                      const doppia::MetricStereoCamera &camera,
                                                      const float ground_plane_prior_pitch,
                                                      const float ground_plane_prior_roll,
                                                      const float ground_plane_prior_height,
                                                      doppia::VideoFromFiles *video_input_p = NULL);
};
}

#endif // EXTENDEDSTIXELWORLDESTIMATORFACTORY_H
