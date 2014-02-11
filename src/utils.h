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

#ifndef STIXEL_UTILS_H
#define STIXEL_UTILS_H

#include <opencv2/opencv.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/program_options.hpp>

namespace stixel_world {
    extern "C" {
        uint8_t waitForKey(uint32_t * time = NULL);
    };

    template <class T>
    inline void opencv2gil(const cv::Mat & imgOpenCV, T & view) {
        for (uint32_t y = 0; y < imgOpenCV.rows; y++) {
            for (uint32_t x = 0; x < imgOpenCV.cols; x++) {
                const cv::Vec3b & pxOCV = imgOpenCV.at<cv::Vec3b>(y, x);
                view(x, y)[0] = (uint8_t)pxOCV[2];
                view(x, y)[1] = (uint8_t)pxOCV[1];
                view(x, y)[2] = (uint8_t)pxOCV[0];
            }
        }
    }
    
    template <class T>
    inline void gil2opencv(const T & view, cv::Mat & imgOpenCV) {    
        imgOpenCV = cv::Mat(view.height(), view.width(), CV_8UC3);
        
        #pragma omp parallel for schedule(static)
        for (uint32_t y = 0; y < imgOpenCV.rows; y++) {
            for (uint32_t x = 0; x < imgOpenCV.cols; x++) {
                cv::Vec3b & pxOCV = imgOpenCV.at<cv::Vec3b>(y, x);
                pxOCV[0] = (uint8_t)view(x, y)[2];
                pxOCV[1] = (uint8_t)view(x, y)[1];
                pxOCV[2] = (uint8_t)view(x, y)[0];
            }
        }
    }
    
    template<class T> 
    void modify_variable_map(std::map<std::string, boost::program_options::variable_value>& vm, const std::string& opt, const T& val) { 
        vm[opt].value() = boost::any(val);
    }
}

// void addLineToPointCloud(const PointType& p1, const PointType& p2, 
//                                           const uint8_t & r, const uint8_t & g, const uint8_t  & b,
//                                           PointCloudTypeExt::Ptr & linesPointCloud, double zOffset) {
//     
//     double dist = pcl::euclideanDistance(p1, p2);
//     
//     const uint32_t nSamples = (uint32_t)(ceil(dist / 0.02));
//     
//     for (uint32_t i = 0; i <= nSamples; i++) {
//         pcl::PointXYZRGB p;
//         p.x = p1.x + ((double)i / nSamples) * (p2.x - p1.x);
//         p.y = p1.y + ((double)i / nSamples) * (p2.y - p1.y);
//         p.z = zOffset;
//         
//         p.r = r;
//         p.g = g;
//         p.b = b;
//         
//         linesPointCloud->push_back(p);
//     } 
//                                           }

#endif