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


#ifndef OFLOWTRACKER_H
#define OFLOWTRACKER_H

#include <opencv2/opencv.hpp>
#include "densetracker.h"
#include "stereo_matching/stixels/Stixel.hpp"
#include <boost/shared_ptr.hpp>
#include <tiff.h>
#include <vector>

// #include <eigen3/Eigen/Sparse>

using namespace doppia;
using namespace std;

namespace stixel_world {

class oFlowTracker
{
public:    
    oFlowTracker();
    
    void compute(const cv::Mat &currImgL, const cv::Mat &currImgR, const stixels_t & stixels);
    
//         typedef vector < stixels3d_t > t_tracker;
//         typedef deque <stixels3d_t> t_historic;
//         t_tracker getTracker() { return m_tracker; }
//         t_historic getHistoric() { return m_stixelsHistoric; }
//         
//         stixels3d_t getLastStixelsAfterTracking();
    
protected:  
    cv::Mat m_lastImgL, m_lastImgR;            // TODO: Just for visualization purposes. Remove once everything is working fine
    boost::shared_ptr<dense_tracker::DenseTracker> m_pDenseTrackerL;
    boost::shared_ptr<dense_tracker::DenseTracker> m_pDenseTrackerR;   // TODO: Am I using this?
    
//     typedef Eigen::SparseMatrix<uint32_t> t_histogram;
    vector < cv::SparseMat > m_histograms;
};

}

#endif // OFLOWTRACKER_H
