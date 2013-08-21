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

#include "fundamentalmatrixestimator.h"

using namespace stixel_world;

FundamentalMatrixEstimator::FundamentalMatrixEstimator()
{

}

FundamentalMatrixEstimator::~FundamentalMatrixEstimator()
{

}

bool FundamentalMatrixEstimator::findF(const cv::Mat& imgLt0, const cv::Mat& imgRt0, const cv::Mat& imgLt1, const cv::Mat& imgRt1, cv::Mat& F)
{
    visualize(imgLt0, imgRt0, imgLt1, imgRt1, vector < vector <cv::Point2d > > ());
    return true;
}

void FundamentalMatrixEstimator::findInitialPoints(const cv::Mat& img, vector< cv::Point2d >& points)
{

}

void FundamentalMatrixEstimator::visualize(const cv::Mat& imgLt0, const cv::Mat& imgRt0, const cv::Mat& imgLt1, const cv::Mat& imgRt1, vector< vector< cv::Point2d > > correspondences)
{
    cv::Mat Lt0, Lt1, Rt0, Rt1;
    
    imgLt0.copyTo(Lt0);
    
    cv::imshow("test", Lt0);
    
    cv::waitKey(0);
}
