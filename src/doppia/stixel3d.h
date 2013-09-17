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

#ifndef STIXEL3D_H
#define STIXEL3D_H

#include "stereo_matching/stixels/Stixel.hpp"
#include <video_input/MetricCamera.hpp>
#include <video_input/MetricStereoCamera.hpp>
#include <opencv2/core/core.hpp>

namespace stixel_world {
class Stixel3d : public doppia::Stixel
{
public:
    Stixel3d(const doppia::Stixel& stixel);
    
    void update3dcoords(const doppia::MetricStereoCamera & camera);
    
    // Bottom in real world coordinates
    cv::Point3d bottom3d;
    // Top in real world coordinates
    cv::Point3d top3d;
    //Depth (avoids recalculating it each time)
    float depth;
    
    // Tells if an stixel is an static obstacle or not
    bool isStatic;
    
    
    template<class T>
    T getBottom2d() { return T(x, bottom_y); }
    template<class T>
    T getTop2d() { return T(x, top_y); }
};

typedef std::vector<Stixel3d> stixels3d_t;

}


#endif // STIXEL3D_H
