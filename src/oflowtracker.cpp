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



#include "oflowtracker.h"

#include <limits.h>

#include "utils.h"
#include <densetracker.h>

#include <omp.h>
#include <boost/foreach.hpp>

namespace stixel_world {

oFlowTracker::oFlowTracker()                                              
{
    m_pDenseTrackerL.reset(new dense_tracker::DenseTracker(dense_tracker::DenseTracker::TRACKING_LEAR));
    m_pDenseTrackerR.reset(new dense_tracker::DenseTracker(dense_tracker::DenseTracker::TRACKING_LEAR));
    
}

// http://cs.brown.edu/courses/csci1290/2011/results/final/psastras/
// http://www.youtube.com/watch?v=0uhZFEhIG-0
//     http://nghiaho.com/?page_id=189
// http://www.inf.ethz.ch/personal/chzach/opensource.html
void oFlowTracker::compute(const cv::Mat& currImgL, const cv::Mat& currImgR, const stixels_t & stixels)
{
    const double & startWallTime = omp_get_wtime();
//     cv::imshow("currImgL", currImgL);
//     cv::imshow("currImgR", currImgR);
    m_pDenseTrackerL->compute(currImgR);
    
    if (m_histograms.size() == 0) {
        m_histograms.resize(stixels.size());
        
        int size[] = {2 * currImgR.rows + 1, 2 * currImgR.cols + 1};
        BOOST_FOREACH(cv::SparseMat & hist, m_histograms) {
            hist.create(2, size, CV_32S);
        }
    }
    
    const cv::Point2i middlePoint(currImgR.cols, currImgR.rows);
    
    
    if (! m_lastImgL.empty()) {
        for (uint32_t x = 0; x < stixels.size(); x++) {
            m_histograms[x].clear();
            
            for (uint32_t y = stixels[x].top_y; y <= stixels[x].bottom_y; y++) {
                const cv::Point2i currPoint(x, y);
                const cv::Point2i & prevPoint = m_pDenseTrackerL->getPrevPoint(currPoint);
                
                if (prevPoint != cv::Point2i(-1, -1)) {
                
                    const cv::Point2i increment = currPoint - prevPoint + middlePoint;
                    
    //                 cout << "Accessing " << increment << endl;
                    m_histograms[x].ref<int32>(increment.y, increment.x) += 1;
    //                 cout << increment << " = " << m_histograms[x].coeffRef(increment.y, increment.x) << endl;
                }
            }
        }
//         cv::imshow("m_lastImgL", m_lastImgL);
//         cv::imshow("m_lastImgR", m_lastImgR);
    
        cout << "Time for oflowtracker:" << __FUNCTION__ << ": " << omp_get_wtime() - startWallTime << endl;
        
        cv::Mat outputR(currImgR);
        for (uint32_t x = 0; x < stixels.size(); x++) {

            
            
            double maxVal = std::numeric_limits<int32_t>::min();
            cv::Point2i maxIdx;
            
            for (cv::SparseMat::iterator it = m_histograms[x].begin(); it != m_histograms[x].end(); it++) {
//                 cout << cv::Point2i(it.node()->idx[1], it.node()->idx[0]) << " = " << it.value<int32>() << endl;
                if (maxVal < it.value<int32>()) {
                    maxIdx = cv::Point2i(it.node()->idx[1], it.node()->idx[0]);
                    maxVal = it.value<int32>();
                    
//                     cout << maxIdx << " = " << maxVal << endl;
                }
            }
//             cout << "*****************************************" << endl;
            const cv::Point2i increment = maxIdx - middlePoint;
            const cv::Point2i currPoint = cv::Point2i(x, stixels[x].bottom_y);
            const cv::Point2i finalPoint = currPoint + increment; // * 10.0;
            
//             cout << "increment " << increment << endl;
//             cout << "currPoint " << currPoint << endl;
//             cout << "finalPoint " << finalPoint << endl;
            
//             cout << x << " -> " << increment << endl;
//             m_histograms[x].maxCoeff();
//             const cv::Point2i increment = 
            
            cv::line(outputR, currPoint, finalPoint, cv::Scalar(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF), 1);
            outputR.at<cv::Vec3b>(stixels[x].top_y, x) = cv::Vec3b(0, 255, 0);
            outputR.at<cv::Vec3b>(stixels[x].bottom_y, x) = cv::Vec3b(0, 255, 0);
//             exit(0);
        }
        cv::imshow("outputL", outputR);
        
        cv::Mat outputFlow(currImgR);
        m_pDenseTrackerL->drawTracks(outputFlow);
        cv::imshow("outputFlow", outputFlow);
        cv::imshow("m_lastImgR", m_lastImgR);
    }
    
    currImgL.copyTo(m_lastImgL);
    currImgR.copyTo(m_lastImgR);

}

}

