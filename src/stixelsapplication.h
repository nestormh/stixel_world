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

#ifndef STIXELSAPPLICATION_H
#define STIXELSAPPLICATION_H

#include <string>

#include <boost/program_options.hpp>

#include "stixel_world_lib.hpp"
#include "video_input/VideoInputFactory.hpp"
#include "stereo_matching/stixels/AbstractStixelWorldEstimator.hpp"

using namespace std;

namespace stixel_world {

class StixelsApplication
{
public:
    StixelsApplication(const string & optionsFile);
    
    void runStixelsApplication();
private:
    boost::program_options::variables_map parseOptionsFile(const string& optionsFile);
    bool iterate();
    void update();
    void visualize();
    
    boost::shared_ptr<doppia::AbstractVideoInput> m_video_input_p;
    boost::shared_ptr<doppia::AbstractStixelWorldEstimator> m_stixel_world_estimator_p;
    
    doppia::AbstractVideoInput::input_image_t m_prevLeftRectified, m_prevRightRectified;
//     boost::gil::rgb8_view_t m_prevLeftRectified, m_prevRightRectified;
//     stixel_world::input_image_const_view_t m_currentLeftRectified, m_currentRightRectified;
//     stixel_world::input_image_const_view_t m_prevLeftRectified, m_prevRightRectified;
    
    boost::shared_ptr<stixels_t> m_currStixels;
    boost::shared_ptr<stixels_t> m_prevStixels;
};

    
}

#endif // STIXELSAPPLICATION_H
