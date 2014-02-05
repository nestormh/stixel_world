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


#ifndef EXTENDEDVIDEOFROMFILES_H
#define EXTENDEDVIDEOFROMFILES_H

#include "video_input/VideoFromFiles.hpp"

// #include "video_input/preprocessing/AbstractPreprocessor.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <boost/thread.hpp>


namespace doppia
{
    
    using boost::shared_ptr;
    
    ///
    /// Loads images from a video stream stored as a set of images.
    /// Supports a preprocessor object for things like unbayering, rectification, etc ...
    ///
    /// Based on Andreas Ess code
    ///
class ExtendedVideoFromFiles : public VideoFromFiles
{
public:
    
    static boost::program_options::options_description get_args_options();
    
    ExtendedVideoFromFiles(const boost::program_options::variables_map &options,
                    const shared_ptr<StereoCameraCalibration> &stereo_calibration_p);
    
    bool next_frame();
    
    bool previous_frame();
protected:
    
//     void read_future_frame_thead();
};
}

#endif // EXTENDEDVIDEOFROMFILES_H
