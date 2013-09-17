/*
    Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>

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


#include "groundestimator.h"
#include "omp.h"

using namespace std;
using namespace stixel_world;

GroundEstimator::GroundEstimator(const Rectification & rectification) : m_rectification(rectification)
{
    m_justHalfImage = true;
    m_yStride = 1;
    m_maxDisparity = 128;
    
    // alpha and v0 as defined in section II of the V-disparity paper of Labayrade, Aubert and Tarel 2002.
    m_stereoAlpha = (
                    m_rectification.getFocalLengthX(0) +
                    m_rectification.getFocalLengthY(0) +
                    m_rectification.getFocalLengthX(1) +
                    m_rectification.getFocalLengthY(1)
                    ) / 4.0;
            
    m_stereoV0 = (
                m_rectification.getFocalCenterY(0) +
                m_rectification.getFocalCenterY(1)
                 ) / 2.0;
}

GroundEstimator::~GroundEstimator()
{

}

void GroundEstimator::setImagePair(const cv::Mat& img1, const cv::Mat& img2)
{
    if (m_justHalfImage) {
        m_left = img1(cv::Range(img1.rows / 2.0, img1.rows), cv::Range::all());
        m_right = img2(cv::Range(img2.rows / 2.0, img2.rows), cv::Range::all());
    } else {
        m_left = img1;
        m_right = img2;
    }

    m_disparity = cv::Mat::zeros(m_left.rows, m_maxDisparity, CV_16UC1);
    m_selectedPoints.clear();
    m_rowWeights.resize(m_disparity.rows);
}

bool GroundEstimator::compute()
{
    static int num_iterations = 0;
    static double cumulated_time = 0;
    
    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();
    
    // compute v_disparity --
    computeVDisparityData();
    
    setPointsWeights(m_pointWeights);
    // compute line --
    estimateGroundPlane();
//     
//     confidence_is_up_to_date = false;
//     
//     // timing ---
//     cumulated_time += omp_get_wtime() - start_wall_time;
//     num_iterations += 1;
//     
//     if((silent_mode == false) and ((num_iterations % num_iterations_for_timing) == 0))
//     {
//         printf("Average FastGroundPlaneEstimator::compute speed  %.2lf [Hz] (in the last %i iterations)\n",
//                num_iterations / cumulated_time, 0/*num_iterations*/ );
//     }
    cout << "Time " << omp_get_wtime() - start_wall_time << endl;
    
    return true;
}

void GroundEstimator::computeVDisparityData()
{
    // for each pixel and each disparity value
    #pragma omp parallel for
    for(uint32_t rowIdx = 0; rowIdx < m_left.rows; rowIdx += m_yStride) {
        computeVDisparityRow(rowIdx);
    }
    
    //TODO: Debug
    cv::Mat visualizePoints = cv::Mat::zeros(m_disparity.rows, m_disparity.cols, CV_8UC3);
    for (uint32_t i = 0; i < m_selectedPoints.size(); i++)
        visualizePoints.at<cv::Vec3b>(m_selectedPoints[i].y, m_selectedPoints[i].x) = cv::Vec3b(0, 0, 255);
    
    cv::namedWindow("vdisparity");
    cv::imshow("vdisparity", m_disparity);
    cv::namedWindow("visualizePoints");
    cv::imshow("visualizePoints", visualizePoints);
    //TODO: End of Debug
    
    //printf("num_points == %i\n", points.size());
    return;
}

inline void GroundEstimator::computeVDisparityRow(const uint32_t& rowIdx)
{
    const int disparityOffset = m_rectification.getDisparityOffsetX();
    const uint16_t costSumSaturation = 5 * 3 * 16; // 5 * number_*of_pixels * levels_by_disparity
    assert(m_maxDisparity <= m_disparity.cols);
    
    uint16_t minCost = std::numeric_limits<uint16_t>::max();
    
    // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
    //const int first_right_x = first_left_x - disparity;
    for(uint32_t d = 0; d < m_maxDisparity; d ++) {
        uint16_t disparityCost = 0;
        
        for (uint32_t iL = d + disparityOffset, iR = 0; iL < m_left.cols; iL++, iR++) {
            const uint16_t cost = sad_cost_uint16(m_left.at<cv::Vec3b>(rowIdx, iL), m_right.at<cv::Vec3b>(rowIdx, iR));
            disparityCost += std::min(cost, costSumSaturation);
        }
        
        // we divide once at the end of the sums
        // this is ok to delay the division because
        // log2(1024*255*3) ~= 20 [bits]
        // so there is no risk of overflow inside 32bits
        disparityCost /= 3;
        m_disparity.at<uint16_t>(rowIdx, d) = disparityCost;
        
        minCost = std::min(disparityCost, minCost);
    } // end of "for each disparity"
    
    // select points to use for ground estimation --
    selectPointsAndWeights(rowIdx, minCost);
    
    return;
}

// compute the raw SAD (without doing a division by the number of channels)
inline uint16_t GroundEstimator::sad_cost_uint16(const cv::Vec3b& pixel_a, const cv::Vec3b& pixel_b)
{
    const int16_t delta_r = pixel_a[0] - pixel_b[0];
    const int16_t delta_g = pixel_a[1] - pixel_b[1];
    const int16_t delta_b = pixel_a[2] - pixel_b[2];
    
    // SAD
    const uint16_t distance = std::abs(delta_r) + std::abs(delta_g) + std::abs(delta_b);
    const uint16_t &cost = distance; // we skip the /3
    
    return cost;
}

inline void GroundEstimator::selectPointsAndWeights(const uint32_t& rowIdx, const uint16_t& minCost)
{
// vector<cv::Point2f> &points, vector<double> &row_weights
    const uint16_t & threshCost = minCost + DELTA_COST;
    uint32_t pointsInRow = 0;
    
    for(uint32_t d = 0; d < m_maxDisparity; d++) {
        if (m_disparity.at<uint16_t>(rowIdx, d) <= threshCost) {
            #pragma omp critical
            {
                m_selectedPoints.push_back( cv::Point2f(d, rowIdx) );
            }
            pointsInRow++;
            
            if(pointsInRow > MAX_POINTS_IN_ROW) {
                // this line is useless, no need to collect more points
                break;
            }
        } else {
           // we discard this point
            continue;
        }
    } // end of "for each disparity"
    
    if(pointsInRow > 0) {
        // rows with less points give more confidence
        m_rowWeights[rowIdx] = 1.0 / (double)pointsInRow;
    }
    
    return;
}

void GroundEstimator::setPointsWeights(Eigen::VectorXf & pointWeights)
{
    pointWeights.resize(m_selectedPoints.size());
    
    for (uint32_t i = 0; i < m_selectedPoints.size(); i++) {
        pointWeights[i] = m_rowWeights[m_selectedPoints[i].y];
    }
        
    return;
}

void GroundEstimator::estimateGroundPlane()
{
    const bool found_ground_plane = findGroundLine(m_vDisparityGroundLine);
    
    /*const float weight = std::max(0.0f, 1.0f - get_confidence());
    
    const float minimum_weight = rejection_threshold;
    //const float minimum_weight = 0.0015;
    
    // retrieve ground plane parameters --
    if((found_ground_plane == true) and (weight >  minimum_weight))
    {
        set_ground_plane_estimate(
            v_disparity_line_to_ground_plane(v_disparity_ground_line), weight);
        
        const bool print_estimated_plane = false;
        if(print_estimated_plane)
        {
            log_debug() << "Found a ground plane with " <<
            "heigth == " << estimated_ground_plane.get_height() << " [meters]"
            " and pitch == " << estimated_ground_plane.get_pitch() * 180 / M_PI << " [degrees]" <<
            std::endl;
            log_debug() << "Ground plane comes from line with " <<
            "origin == " << v_disparity_ground_line.origin()(0) << " [pixels]"
            " and direction == " << v_disparity_ground_line.direction()(0) << " [-]" <<
            std::endl;
        }
    }
    else
    {
        num_ground_plane_estimation_failures += 1;
        
        // in case this happened during the first call
        // we set the v_disparity_ground_line using the current ground plane estimate
        v_disparity_ground_line = ground_plane_to_v_disparity_line( get_ground_plane() );
        
        static int num_ground_warnings = 0;
        //const int max_num_ground_warnings = 1000;
        //const int max_num_ground_warnings = 50;
        const int max_num_ground_warnings = 25;
        
        if(num_ground_warnings < max_num_ground_warnings)
        {
            log_warning() << "Did not find a ground plane, keeping previous estimate." << std::endl;
            num_ground_warnings += 1;
        }
        else if(num_ground_warnings == max_num_ground_warnings)
        {
            log_warning() << "Warned too many times about problems finding ground plane, going silent." << std::endl;
            num_ground_warnings += 1;
        }
        else
        {
            // we do nothing
        }
        
        const float weight = 1.0;
        // we keep previous estimate
        set_ground_plane_estimate(estimated_ground_plane, weight);
        
        
        //        const bool save_failures_v_disparity_image = false;
        //        if(save_failures_v_disparity_image)
        //        {
        //            const std::string filename = boost::str(
        //                        boost::format("failure_v_disparity_%i.png") % num_ground_plane_estimation_failures );
        //            log_info() << "Created image " << filename << std::endl;
        //            boost::gil::png_write_view(filename, v_disparity_image_view);
        //        }
    }
    
    return;*/
}

bool GroundEstimator::findGroundLine(line_t &groundLine) {
    // find the most likely plane (line) in the v-disparity image ---
    vector<line_t> foundLines;
    bool foundGroundPlane = false;
    
    // we correct the origin of our estimate lines
    // since we computed them using the lower half of the image
    const int originOffset = m_left.rows;
    
    const line_t linePrior = groundPlaneToVDisparityLine(m_estimatedGroundPlane);
    
    line_t linePriorWithOffset = linePrior;
    linePriorWithOffset.origin()(0) -= originOffset;
    cout << "m_estimatedGroundPlane " << m_estimatedGroundPlane.get_pitch() << ", " << m_estimatedGroundPlane.get_height() << endl;
    cout << "linePrior " << linePrior.direction() << endl;
    
    m_pIrlsLinesDetector->set_initial_estimate(linePriorWithOffset);
    
    exit(0);
    (*m_pIrlsLinesDetector)(m_selectedPoints, m_pointWeights, foundLines);
    
    printf("irls_lines_detector_p found %zi lines\n", foundLines.size());
    
    /*BOOST_FOREACH(line_t &line, found_lines)
    {
        line.origin()(0) += origin_offset;
    }
    
    // given two bounding lines we verify the x=0 line and the y=max_y line
    // this checks bound quite well the desired ground line
    
    const float direction_fraction = 1.5; // FIXME hardcoded value
    const float max_line_direction = prior_max_v_disparity_line.direction()(0)*direction_fraction,
    min_line_direction = prior_min_v_disparity_line.direction()(0)/direction_fraction;
    
    const float max_line_y0 = prior_max_v_disparity_line.origin()(0),
    min_line_y0 = prior_min_v_disparity_line.origin()(0);
    
    const float min_y0 = max_line_y0, max_y0 = min_line_y0;
    
    const float y_intercept = input_left_view.height();
    const float max_x_intercept = (y_intercept - max_line_y0) / max_line_direction,
    min_x_intercept = (y_intercept - min_line_y0) / min_line_direction;
    
    assert(min_y0 < max_y0);
    assert(min_x_intercept < max_x_intercept);
    
    BOOST_FOREACH(line_t t_ground_line, found_lines)
    {
        const float t_y0 = t_ground_line.origin()(0);
        const float t_direction =  t_ground_line.direction()(0);
        const float t_x_intercept = (y_intercept - t_y0) / t_direction;
        
        const bool print_xy_check = false;
        if(print_xy_check)
        {
            printf("prior origin == %.3f, direction == %.3f\n",
                   line_prior.origin()(0), line_prior.direction()(0));
            
            printf("line origin == %.3f, direction == %.3f\n",
                   t_ground_line.origin()(0), t_ground_line.direction()(0));
            
            printf("max_y0 == %.3f, t_y0 == %.3f, min_y0 = %.3f\n",
                   max_y0, t_y0, min_y0);
            
            printf("max_x_intercept == %.3f, t_x_intercept == %.3f, min_x_intercept = %.3f\n",
                   max_x_intercept, t_x_intercept, min_x_intercept);
            
            printf("max_line_direction == %.3f, t_direction == %.3f, min_line_direction = %.3f\n",
                   max_line_direction, t_direction, min_line_direction);
        }
        
        if(t_y0 <= max_y0 and t_y0 >= min_y0 and
            t_x_intercept <= max_x_intercept and t_x_intercept >= min_x_intercept and
            t_direction <= max_line_direction and t_direction >= min_line_direction )
        {
            ground_line = t_ground_line;
            found_ground_plane = true;
            break;
        }
        else
        {
            continue;
        }
        
    } // end of "for each found line"*/
    
    return foundGroundPlane;
}

GroundEstimator::line_t GroundEstimator::groundPlaneToVDisparityLine(const doppia::GroundPlane &groundPlane) {
    const float & theta = -groundPlane.get_pitch();
    const float & heigth = groundPlane.get_height();
    
    cout << "theta " << theta << endl;
    cout << "heigth " << heigth << endl;
    
    line_t line;
    
    // based on equations 10 and 11 from V-disparity paper of Labayrade, Aubert and Tarel 2002.
    const float v_origin = m_stereoV0 - m_stereoAlpha * std::tan(theta);
    const float c_r = m_rectification.getBaseline() * cos(theta) / heigth;
    
    line.origin()(0) = v_origin;
    line.direction()(0) = 1.0 / c_r;
    
    
    exit(0);
    
    return line;
}