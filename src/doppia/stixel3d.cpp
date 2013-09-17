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

#include "stixel3d.h"

using namespace stixel_world;

Stixel3d::Stixel3d(const doppia::Stixel& stixel)
{
    width = stixel.width;
    x = stixel.x;
    bottom_y = stixel.bottom_y; 
    top_y = stixel.top_y;
    default_height_value = stixel.default_height_value;
    disparity = stixel.disparity;
    backward_delta_x = stixel.backward_delta_x;
    valid_backward_delta_x = stixel.valid_backward_delta_x;
    backward_width = stixel.backward_width; 
    
    isStatic = false;
    
}

void Stixel3d::update3dcoords(const doppia::MetricStereoCamera& camera)
{
    Eigen::Vector2f bottom2d, top2d;
    bottom2d << x, bottom_y;
    top2d << x, top_y;
    if (disparity > 0.0f)
        depth = camera.disparity_to_depth(disparity );
    const Eigen::Vector3f & bottom3dvector = camera.get_left_camera().back_project_2d_point_to_3d(bottom2d, depth);
    const Eigen::Vector3f & top3dvector = camera.get_left_camera().back_project_2d_point_to_3d(top2d, depth);
    
    bottom3d = cv::Point3d(bottom3dvector[0], bottom3dvector[1], bottom3dvector[2]);
    top3d = cv::Point3d(top3dvector[0], top3dvector[1], top3dvector[2]);
}
