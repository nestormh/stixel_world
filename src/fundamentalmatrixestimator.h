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

#ifndef FUNDAMENTALMATRIXESTIMATOR_H
#define FUNDAMENTALMATRIXESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

namespace stixel_world {
    
class FundamentalMatrixEstimator
{

public:
    FundamentalMatrixEstimator();
    ~FundamentalMatrixEstimator();
    
    bool findF(const cv::Mat & imgLt0, const cv::Mat & imgRt0, 
               const cv::Mat & imgLt1, const cv::Mat & imgRt1, 
               cv::Mat & F);
private:
    void findInitialPoints(const cv::Mat & img, vector<cv::Point2d> & points);
    void visualize(const cv::Mat & imgLt0, const cv::Mat & imgRt0, 
                   const cv::Mat & imgLt1, const cv::Mat & imgRt1, 
                   vector < vector < cv::Point2d > > correspondences);
};

}
#endif // FUNDAMENTALMATRIXESTIMATOR_H
