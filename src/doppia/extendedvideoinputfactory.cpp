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

#include "extendedvideoinputfactory.h"
#include "extendedvideofromfiles.h"

#include "video_input/VideoInputFactory.hpp"

#include "video_input/VideoFromFiles.hpp"
#include "video_input/preprocessing/CpuPreprocessor.hpp"
// #include "video_input/calibration/StereoCameraCalibration.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>

#include "helpers/get_option_value.hpp"
// #include "helpers/get_section_options.hpp"

    
namespace doppia
{
    
using namespace std;
using namespace boost::program_options;

options_description
ExtendedVideoInputFactory::get_args_options()
{
    options_description desc("VideoInputFactory options");
    
    desc.add_options()
    
    ("video_input.source", value<string>()->default_value("directory"),
                                                            "video input source: directory, movie or camera")
    
    ("video_input.calibration_filename", value<string>(),
        "filename protocol buffer text description of the stereo rig calibration. See calibration.proto for mor details")
    
    ;
    
    desc.add(AbstractVideoInput::get_args_options());
    desc.add(VideoFromFiles::get_args_options());
    desc.add(AbstractPreprocessor::get_args_options());
    desc.add(CpuPreprocessor::get_args_options());
    
    //        desc.add(get_section_options("video_input", "AbstractVideoInput options", AbstractVideoInput::get_args_options()));
    //        desc.add(get_section_options("video_input", "VideoFromFiles options", VideoFromFiles::get_args_options()));
    //        desc.add(get_section_options("video_input", "AbstractPreprocessor options", AbstractPreprocessor::get_args_options()));
    //   desc.add(get_section_options("video_input", "CpuPreprocessor options", CpuPreprocessor::get_args_options()));
    
    return desc;
}

extern shared_ptr<AbstractPreprocessor> new_preprocessor_instance(const variables_map &options, AbstractVideoInput &video_input);
// shared_ptr<AbstractPreprocessor> new_preprocessor_instance(const variables_map &options, AbstractVideoInput &video_input)
// {
//     shared_ptr<AbstractPreprocessor> preprocess_p;
//     preprocess_p.reset(new CpuPreprocessor(video_input.get_left_image().dimensions(),
//                                             video_input.get_stereo_calibration(),
//                                             options));
//     return preprocess_p;
// }

AbstractVideoInput*
ExtendedVideoInputFactory::new_instance(const variables_map &options)
{
    // create the stereo matcher instance
    const string source = get_option_value<std::string>(options, "video_input.source");
    const string calibration_filename = get_option_value<std::string>(options, "video_input.calibration_filename");
    
    // the calibration object is temporary, used only to precompute data inside the CpuPreprocessor
    const shared_ptr<StereoCameraCalibration> stereo_calibration_p(new StereoCameraCalibration(calibration_filename));
    
    AbstractVideoInput* video_source_p = NULL;
    if (source.compare("directory") == 0)
    {
        VideoFromFiles * const video_from_files_p = new VideoFromFiles(options, stereo_calibration_p);
        video_source_p = video_from_files_p;
        
        video_from_files_p->set_preprocessor(new_preprocessor_instance(options, *video_source_p));
    }
    else if (source.compare("movie") == 0)
    {
        throw std::runtime_error("movie video input is not yet implemented");
    }
    else if (source.compare("camera") == 0)
    {
        throw std::runtime_error("camera video input is not yet implemented");
    } else if (source.compare("directory_skip") == 0)
    {
        ExtendedVideoFromFiles * const video_from_files_p = new ExtendedVideoFromFiles(options, stereo_calibration_p);
        video_source_p = video_from_files_p;
        
        video_from_files_p->set_preprocessor(new_preprocessor_instance(options, *video_source_p));
    }
    else
    {
        throw std::runtime_error("Unknown 'video_input.source' value");
    }
    
    
    return video_source_p;
}
}