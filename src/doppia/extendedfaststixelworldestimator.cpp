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


#include "extendedfaststixelworldestimator.h"

#include "extendedfastgroundplaneestimator.h"

#include "video_input/MetricStereoCamera.hpp"

#include <iostream>

using namespace stixel_world;
using namespace std;

ExtendedFastStixelWorldEstimator::ExtendedFastStixelWorldEstimator(const boost::program_options::variables_map &options,
                                                                   const doppia::AbstractVideoInput::dimensions_t &input_dimensions,
                                                                   const doppia::MetricStereoCamera &camera,
                                                                   const doppia::GroundPlane &ground_plane_prior) :
                                                                    doppia::FastStixelWorldEstimator(options, input_dimensions, camera, ground_plane_prior)
{
    ground_plane_estimator_p.reset(new ExtendedFastGroundPlaneEstimator(options, camera.get_calibration()));
}

ExtendedFastStixelWorldEstimator::~ExtendedFastStixelWorldEstimator()
{

}

