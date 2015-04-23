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


#include "stereo_matching/stixels/StixelWorldEstimatorFactory.hpp"

#include "extendedfaststixelworldestimator.h"
#include "extendedfastgroundplaneestimator.h"

#include "stereo_matching/stixels/StixelWorldEstimator.hpp"
#include "stereo_matching/stixels/FastStixelWorldEstimator.hpp"

#include "stereo_matching/stixels/StixelsEstimator.hpp"
#include "stereo_matching/stixels/StixelsEstimatorWithHeightEstimation.hpp"
#include "stereo_matching/stixels/ImagePlaneStixelsEstimator.hpp"

#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"
#include "image_processing/IrlsLinesDetector.hpp"

#include "video_input/VideoFromFiles.hpp"

#include "helpers/get_option_value.hpp"

#include <string>

#include <iostream>

#include "extendedstixelworldestimatorfactory.h"

namespace stixel_world {

using namespace std;
using namespace boost;
using namespace program_options;
using namespace doppia;

options_description
ExtendedStixelWorldEstimatorFactory::get_args_options()
{
    options_description desc("StixelWorldEstimatorFactory options");
    
    desc.add_options()
    ("stixel_world.method", value<string>()->default_value("fast_uv"),
                                                           "stixels world estimation methods: multilayer, fast_uv, fast, not_fast. "
                                                           "fast_uv indicates that the stixels will be directly estimated in the image plane, "
                                                           "instead of in the disparity space (fast).")
    ;
    
    //desc.add(AbstractStixelWorldEstimator::get_args_options());
    desc.add(StixelWorldEstimator::get_args_options());
    desc.add(FastStixelWorldEstimator::get_args_options());
    desc.add(ExtendedFastStixelWorldEstimator::get_args_options());
    
    desc.add(StixelsEstimator::get_args_options());
    desc.add(StixelsEstimatorWithHeightEstimation::get_args_options());
    desc.add(ImagePlaneStixelsEstimator::get_args_options());
    
    desc.add(BaseGroundPlaneEstimator::get_args_options());
    desc.add(GroundPlaneEstimator::get_args_options());
    desc.add(FastGroundPlaneEstimator::get_args_options());
    desc.add(ExtendedFastGroundPlaneEstimator::get_args_options());
    desc.add(IrlsLinesDetector::get_args_options());
    
    return desc;
}


AbstractStixelWorldEstimator*
ExtendedStixelWorldEstimatorFactory::new_instance(const variables_map &options,
                                          AbstractVideoInput &video_input)
{
    const AbstractVideoInput::dimensions_t &input_dimensions = video_input.get_left_image().dimensions();
    const MetricStereoCamera &camera = video_input.get_metric_camera();
    
    cout << "VideoInput" << endl;
    VideoFromFiles *video_input_p = dynamic_cast<VideoFromFiles*>(&video_input);
    
    return new_instance(options, input_dimensions, camera,
                        video_input.camera_pitch, video_input.camera_roll, video_input.camera_height,
                        video_input_p);
}


AbstractStixelWorldEstimator*
ExtendedStixelWorldEstimatorFactory::new_instance(const boost::program_options::variables_map &options,
                                          const AbstractVideoInput::dimensions_t &input_dimensions,
                                          const MetricStereoCamera &camera,
                                          const float ground_plane_prior_pitch,
                                          const float ground_plane_prior_roll,
                                          const float ground_plane_prior_height,
                                          VideoFromFiles *video_input_p)
{
    // create the stixel_world_estimator instance
    const string method = get_option_value<string>(options, "stixel_world.method");
    
    if(video_input_p == NULL)
    {
        throw std::invalid_argument("video_input_p is NULL!!!");
    }
    
    GroundPlane ground_plane_prior;
    ground_plane_prior.set_from_metric_units(
        ground_plane_prior_pitch, ground_plane_prior_roll, ground_plane_prior_height);
    
    AbstractStixelWorldEstimator* stixel_world_estimator_p = NULL;
    
    if (method.empty() or (method.compare("multilayer") == 0)) {
        stixel_world_estimator_p = new FastStixelWorldEstimator(options,
                                                                          input_dimensions,
                                                                          camera,
                                                                          ground_plane_prior);
    } else if ((method.compare("fast") == 0) or (method.compare("fast_uv") == 0))
    {
        stixel_world_estimator_p = new ExtendedFastStixelWorldEstimator(options,
                                                                input_dimensions,
                                                                camera,
                                                                ground_plane_prior);
    }
    else if ( method.compare("not_fast") == 0)
    {
        if(video_input_p == NULL)
        {
            throw std::invalid_argument("When using stixel_world.method == not_fast "
            "StixelWorldEstimatorFactory only support VideoFromFiles video input");
        }
        
        stixel_world_estimator_p = new StixelWorldEstimator(options,
                                                            input_dimensions,
                                                            camera, video_input_p->get_preprocessor(),
                                                            ground_plane_prior);
    } else
    {
        printf("StixelWorldEstimatorFactory received stixel_world.method value == %s\n", method.c_str());
        throw std::runtime_error("Unknown 'stixel_world.method' value");
    }
    
    
    return stixel_world_estimator_p;
}

}