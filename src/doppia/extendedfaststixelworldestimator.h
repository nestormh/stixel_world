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


#ifndef EXTENDEDFASTSTIXELWORLDESTIMATOR_H
#define EXTENDEDFASTSTIXELWORLDESTIMATOR_H

#include "stereo_matching/stixels/FastStixelWorldEstimator.hpp"

namespace stixel_world {

class ExtendedFastStixelWorldEstimator : public doppia::FastStixelWorldEstimator
{

public:
    ExtendedFastStixelWorldEstimator(const boost::program_options::variables_map &options,
                                     const doppia::AbstractVideoInput::dimensions_t &input_dimensions,
                                     const doppia::MetricStereoCamera &camera,
                                     const doppia::GroundPlane &ground_plane_prior);
    ~ExtendedFastStixelWorldEstimator();
    
    
};

}

#endif // EXTENDEDFASTSTIXELWORLDESTIMATOR_H
