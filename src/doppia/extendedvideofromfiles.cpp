/*
    Copyright 2014 Néstor Morales Hernández <email>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/


#include "extendedvideofromfiles.h"

// #include "video_input/VideoFromFiles.hpp"

// #include "video_input/calibration/StereoCameraCalibration.hpp"

#include "helpers/get_option_value.hpp"

// #include <limits>
// 
// #include <boost/filesystem.hpp>
// 
// #include <boost/gil/image_view.hpp>
// #include <boost/gil/extension/io/png_io.hpp>
// 
// #include <omp.h>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>

#include "helpers/get_option_value.hpp"
#include "helpers/get_section_options.hpp"

#include <string>

using namespace std;
using namespace boost;

using namespace boost::program_options;

namespace doppia
{
    
// options_description 
boost::program_options::options_description
ExtendedVideoFromFiles::get_args_options()
{
    boost::program_options::options_description desc("VideoFromFiles options");
    
    desc.add_options()
    
    // @param mask string containing directory and filename, except
    // an %d for sprintf to be replaced by frame number, e.g. image_%08d.pgm
    
    ("video_input.left_filename_mask",
     boost::program_options::value<string>(),
        "sprintf mask for left image files input. Will receive the frame number as input. Example: the_directory/left_%05d.png")
    
    ("video_input.right_filename_mask",
     boost::program_options::value<string>(),
        "sprintf mask for right image files input. Will receive the frame number as input. Example: the_directory/right_%05d.png")
    
    ("video_input.frame_rate",
     boost::program_options::value<int>()->default_value(15), "video input frame rate")
    
    ("video_input.frame_width",
     boost::program_options::value<int>()->default_value(640), "video input frame width in pixels")
    
    ("video_input.frame_height",
     boost::program_options::value<int>()->default_value(480), "video input frame height in pixels" )
    
    ("video_input.start_frame",
     boost::program_options::value<int>()->default_value(0), "first image to read")
    
    ("video_input.end_frame",
     boost::program_options::value<int>(), "last image to read, if omited will read all files matching the masks")
    ;
    
    
    return desc;
}

ExtendedVideoFromFiles::ExtendedVideoFromFiles(const program_options::variables_map &options,
                               const shared_ptr<StereoCameraCalibration> &stereo_calibration_p)
                    : VideoFromFiles(options, stereo_calibration_p)
{

}
    
/// Advance in stream, return true if successful
bool ExtendedVideoFromFiles::next_frame()
{
    return this->set_frame(current_frame_number + 10);
}

/// Go back in stream
bool ExtendedVideoFromFiles::previous_frame()
{
    return this->set_frame(current_frame_number - 10);
}
    
}

