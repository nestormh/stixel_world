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


#include "stixelsapplicationros.h"
#include </home/nestor/Dropbox/KULeuven/projects/StixelWorld/src/doppia/stixel3d.h>
#include <doppia/stixel3d.h>

#include "doppia/extendedstixelworldestimatorfactory.h"

#include "doppia/extendedvideoinputfactory.h"

#include "utils.h"
#include "fundamentalmatrixestimator.h"

#include <boost/filesystem.hpp>
#include <boost/concept_check.hpp>
#include <boost/graph/graph_concepts.hpp>

#include <fstream>
#include <eigen3/Eigen/src/Core/Matrix.h>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <omp.h>
#include <opencv2/core/core.hpp>

#include "elas.h"

#define CAMERA_FRAME_ID "left_cam"

using namespace stixel_world;

namespace stixel_world_ros {

StixelsApplicationROS::StixelsApplicationROS(const string& optionsFile)
{
    m_options = parseOptionsFile(optionsFile);
    
    mp_video_input.reset(doppia::ExtendedVideoInputFactory::new_instance(m_options));
    
    if(not mp_video_input)
    {
        throw std::invalid_argument("Failed to initialize a video input module. "
        "No images to read, nothing to compute.");
    }

    m_initialFrame = mp_video_input->get_current_frame_number() + 1;
    
    mp_stixel_world_estimator.reset(StixelWorldEstimatorFactory::new_instance(m_options, *mp_video_input));
    mp_prevStixels.reset(new stixels_t);
    mp_stixel_motion_evaluator.reset(new MotionEvaluation(m_options));
    
    m_doPolarCalib = false;
    
    // ROS parameters
    ros::NodeHandle nh("~");
    bool m_useGraph, m_useCostMatrix, m_useObjects, twoLevelsTracking;
    double m_SADFactor, m_heightFactor, m_polarDistFactor, m_polarSADFactor, m_histBatFactor;

    nh.param("useGraph", m_useGraph, true);
    nh.param("useCostMatrix", m_useCostMatrix, true);
    nh.param("useObjects", m_useObjects, true);
    nh.param("twoLevelsTracking", twoLevelsTracking, true);
    
    nh.param("SADFactor", m_SADFactor, 0.0);
    nh.param("heightFactor", m_heightFactor, 0.0);
    nh.param("polarDistFactor", m_polarDistFactor, 0.0);
    nh.param("polarSADFactor", m_polarSADFactor, 0.0);
    nh.param("histBatFactor", m_histBatFactor, 0.0);
    
    nh.param("increment", m_increment, 1);
    
    if (twoLevelsTracking) {
        m_useCostMatrix = true;
        m_useObjects = true;
    }
    
    if ((m_useObjects) || (m_polarDistFactor != 0.0) || (m_polarSADFactor != 0.0)) {
        m_doPolarCalib = true;
    }
    
//     m_doPolarCalib = true;
    
    // TODO: Parameterize
    m_frameBufferLength = 2;
    
    cout << "m_useGraph " << m_useGraph << endl;
    cout << "m_useCostMatrix " << m_useCostMatrix << endl;
    cout << "m_useObjects " << m_useObjects << endl;
    cout << "m_SADFactor " << m_SADFactor << endl;
    cout << "heightFactor " << m_heightFactor << endl;
    cout << "m_polarDistFactor " << m_polarDistFactor << endl;
    cout << "m_polarSADFactor " << m_polarSADFactor << endl;
    cout << "m_histBatFactor " << m_histBatFactor << endl;
    cout << "twoLevelsTracking " << twoLevelsTracking << endl;
    cout << "m_doPolarCalib " << m_doPolarCalib << endl;
    cout << "***********************" << endl;
    
    // ROS publishers / suscribers
    m_pointCloudPub = nh.advertise<sensor_msgs::PointCloud2> ("pointCloudStixels", 1);
    m_fakePointCloudPub = nh.advertise<sensor_msgs::PointCloud2> ("fakePointCloud", 1);
    m_stereoPointCloudPub = nh.advertise<sensor_msgs::PointCloud2> ("stereoPointCloud", 1);
    
    m_clockPub = nh.advertise<rosgraph_msgs::Clock> ("/clock", 1);
    
    image_transport::ImageTransport it(nh);
    m_leftImgPub = it.advertise("left/image", 1);
    m_rightImgPub = it.advertise("right/image", 1);
    
    m_leftInfoPub = nh.advertise<sensor_msgs::CameraInfo>("left/camera_info", 1);
    m_righttInfoPub = nh.advertise<sensor_msgs::CameraInfo>("right/camera_info", 1);
    
    // TODO: Parameterize
    string leftCalibFileName = "/local/imaged/stixels/bahnhof/left_calib.yaml";
    string leftCameraName = "left_camera";
    camera_calibration_parsers::readCalibrationYml(leftCalibFileName, leftCameraName, m_leftCameraInfo);
    
    string rightCalibFileName = "/local/imaged/stixels/bahnhof/right_calib.yaml";
    string rightCameraName = "right_camera";
    camera_calibration_parsers::readCalibrationYml(rightCalibFileName, rightCameraName, m_rightCameraInfo);
    
//     NOTE: This is just for fast tuning of the motion estimators
//     mp_stixels_tests.resize(6);
//     for (uint32_t i = 0; i < mp_stixels_tests.size(); i++) {
//         mp_stixels_tests[i].reset( 
//             new StixelsTracker( m_options, mp_video_input->get_metric_camera(), 
//                                 mp_stixel_world_estimator->get_stixel_width(),
//                                 mp_polarCalibration) );
//     }
//     mp_stixels_tests[0]->set_motion_cost_factors(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, false);
//     mp_stixels_tests[1]->set_motion_cost_factors(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, false);
//     mp_stixels_tests[2]->set_motion_cost_factors(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, false);
//     mp_stixels_tests[3]->set_motion_cost_factors(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, true);
//     mp_stixels_tests[4]->set_motion_cost_factors(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, true);
//     mp_stixels_tests[5]->set_motion_cost_factors(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, true);
//     
//     mp_stixel_motion_evaluator->addStixelMotionEstimator(mp_stixel_world_estimator, mp_stixels_tests[0]);
//     mp_stixel_motion_evaluator->addStixelMotionEstimator(mp_stixel_world_estimator, mp_stixels_tests[1]);
//     mp_stixel_motion_evaluator->addStixelMotionEstimator(mp_stixel_world_estimator, mp_stixels_tests[2]);
//     mp_stixel_motion_evaluator->addStixelMotionEstimator(mp_stixel_world_estimator, mp_stixels_tests[3]);
//     mp_stixel_motion_evaluator->addStixelMotionEstimator(mp_stixel_world_estimator, mp_stixels_tests[4]);
//     mp_stixel_motion_evaluator->addStixelMotionEstimator(mp_stixel_world_estimator, mp_stixels_tests[5]);
// end of NOTE
    
    if (mp_stixels_tests.size() == 0) {
//         m_doPolarCalib = true;
        if (m_doPolarCalib)
            mp_polarCalibration.reset(new PolarCalibration());
        mp_stixel_motion_estimator.reset( 
        new StixelsTracker( m_options, mp_video_input->get_metric_camera(), 
                            mp_stixel_world_estimator->get_stixel_width(),
                            mp_polarCalibration) );
        
//         mp_stixel_motion_estimator->set_motion_cost_factors(0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, true);
        mp_stixel_motion_estimator->set_motion_cost_factors(m_SADFactor, m_heightFactor, m_polarDistFactor, 
                                                            m_polarSADFactor, 0.0f, m_histBatFactor, 
                                                            m_useGraph, m_useCostMatrix, m_useObjects,
                                                            twoLevelsTracking);
        mp_stixel_motion_evaluator->addStixelMotionEstimator(mp_stixel_world_estimator, mp_stixel_motion_estimator);
//         mp_stixel_oflow_motion_estimator.reset(new oFlowTracker());
        
    } else {
        mp_stixel_motion_estimator = mp_stixels_tests[0];
    }
    
    m_waitTime = 0;
    
    m_accTime = 0.0;
    
    m_firstIteration = true;
        
    return;
}

boost::program_options::variables_map StixelsApplicationROS::parseOptionsFile(const string& optionsFile) 
{

    boost::program_options::variables_map options;
    if (optionsFile.empty() == false)
    {
        boost::filesystem::path configurationFilePath(optionsFile);
        if(boost::filesystem::exists(configurationFilePath) == false)
        {
            cout << "\033[1;31mCould not find the configuration file:\033[0m "
                 << configurationFilePath << endl;
            return options;
        }

//         init_stixel_world(configurationFilePath);
        boost::program_options::options_description desc;
        get_options_description(desc);
        desc.add(StixelsTracker::get_args_options());
        desc.add(MotionEvaluation::get_args_options());
        
//         desc.add_options()
//         ("save_stixels",
//          program_options::value<bool>()->default_value(false),
//          "save the estimated stixels in a data sequence file")
//         
//         ("save_ground_plane_corridor",
//          program_options::value<bool>()->default_value(false),
//          "save the estimated expected bottom and top of objects in the data sequence file")
//         
//         ("gui.disabled",
//          program_options::value<bool>()->default_value(false),
//          "if true, no user interface will be presented")
//         
//         ("silent_mode",
//          program_options::value<bool>()->default_value(false),
//          "if true, no status information will be printed at run time (use this for speed benchmarking)")
//         
//         ("stixel_world.motion",
//          boost::program_options::value<bool>()->default_value(false),
//          "if true the stixels motion will be estimated")
        
//         ;
        
        printf("Going to parse the configuration file: %s\n", configurationFilePath.c_str());

        try
        {
            fstream configuration_file;
            configuration_file.open(configurationFilePath.c_str(), fstream::in);
            boost::program_options::store(boost::program_options::parse_config_file(configuration_file, desc), options);
            configuration_file.close();
        }
        catch (...)
        {
            cout << "\033[1;31mError parsing the configuration file named:\033[0m "
            << configurationFilePath << endl;
            cout << desc << endl;
            throw;
        }

        cout << "Parsed the configuration file " << configurationFilePath << std::endl;
    }
    return options;   
}

void StixelsApplicationROS::runStixelsApplication()
{
//     cv::namedWindow("output");
//     cv::moveWindow("output", 1366, 0);
    
    
//     cv::namedWindow("polar");
//     if (mp_stixels_tests.size() == 0)
//         cv::moveWindow("polar", 2646, 0);
    
//     cv::namedWindow("denseTrack");
//     cv::moveWindow("denseTrack", 1366, 480);
    
//     cv::namedWindow("polarTrack");
//     cv::moveWindow("polarTrack", 1366, 0);
    
    rosgraph_msgs::Clock clockMsg;
    m_accTime = 0.0;
    clockMsg.clock = ros::Time(m_accTime);
    m_clockPub.publish(clockMsg);
    ros::spinOnce();
    
    m_prevLeftRectified = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_left_image().dimensions());
    m_prevRightRectified = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_right_image().dimensions());
    
    double startWallTime = omp_get_wtime();
    while (iterate()) {
        visualize();
        publishROS();
//         publishStixels();
        publishStixelsInObjects();
        update();
        cout << "Time for " << __FUNCTION__ << ": " << omp_get_wtime() - startWallTime << endl;
        startWallTime = omp_get_wtime();
        cout << "********************************" << endl;
    }
}

void StixelsApplicationROS::update()
{

    const double & startWallTime = omp_get_wtime();
    
    // Updating the rectified images
    const stixel_world::input_image_const_view_t & currLeft = mp_video_input->get_left_image();
    const stixel_world::input_image_const_view_t & currRight = mp_video_input->get_right_image();
    
    boost::gil::copy_pixels(currLeft, boost::gil::view(m_prevLeftRectified));
    boost::gil::copy_pixels(currRight, boost::gil::view(m_prevRightRectified));
    
//     doppia::AbstractVideoInput::input_image_t bufferImgL, bufferImgR;
    doppia::AbstractVideoInput::input_image_t bufferImgL = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_left_image().dimensions());
    doppia::AbstractVideoInput::input_image_t bufferImgR = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_right_image().dimensions());
    boost::gil::copy_pixels(currLeft, boost::gil::view(bufferImgL));
    boost::gil::copy_pixels(currRight, boost::gil::view(bufferImgR));
    
    m_frameBufferLeft.push_back(bufferImgL);
    m_frameBufferRight.push_back(bufferImgR);
    
    if (m_frameBufferLeft.size() > m_frameBufferLength)
        m_frameBufferLeft.pop_front();
    if (m_frameBufferRight.size() > m_frameBufferLength)
        m_frameBufferRight.pop_front();
    
    // TODO: Use again when speed information is needed
    if (mp_stixel_motion_estimator)
        mp_stixel_motion_estimator->set_estimated_stixels(mp_stixel_world_estimator->get_stixels());
    
//     for (uint32_t i = 0; i < mp_stixels_tests.size(); i++)
//         mp_stixels_tests[i]->set_estimated_stixels(mp_stixel_world_estimator->get_stixels());
    
    // Updating the stixels
    mp_prevStixels->resize(mp_stixel_world_estimator->get_stixels().size());
    std::copy(mp_stixel_world_estimator->get_stixels().begin(), 
                mp_stixel_world_estimator->get_stixels().end(), 
                mp_prevStixels->begin());
    
    cout << "Time for " << __FUNCTION__ << ": " << omp_get_wtime() - startWallTime << endl;
}

bool StixelsApplicationROS::iterate()
{
    const double & startWallTime = omp_get_wtime();
    
    if ((mp_video_input->get_current_frame_number() == mp_video_input->get_number_of_frames()) || (! mp_video_input->next_frame()))
        return false;
    
//     if (m_frameBufferLeft.size() < m_frameBufferLength)
//         return true;
    
    cout << "Frame " << mp_video_input->get_current_frame_number() << endl;
    
    doppia::AbstractVideoInput::input_image_view_t
                        left_view(mp_video_input->get_left_image()),
                        right_view(mp_video_input->get_right_image());  
                        
    gil2opencv(mp_video_input->get_left_image(), m_currLeft);
    gil2opencv(mp_video_input->get_right_image(), m_currRight);
    mp_stixel_world_estimator->set_rectified_images_pair(left_view, right_view);
    mp_stixel_world_estimator->compute();
    
    if (! rectifyPolar()) {
//         TODO: Do something in this case
        return true;
    }

    // TODO: Use again when speed information is needed
    if (mp_stixel_motion_estimator) {
        mp_stixel_motion_estimator->set_new_rectified_image(left_view);
        mp_stixel_motion_estimator->updateDenseTracker(m_currLeft);
        mp_stixel_motion_estimator->set_estimated_stixels(mp_stixel_world_estimator->get_stixels());
        
//         if(mp_video_input->get_current_frame_number() > m_initialFrame - 10)
        if (!m_firstIteration)
            mp_stixel_motion_estimator->compute();
        else
            m_firstIteration = false;
    }
    
    if (mp_stixel_oflow_motion_estimator) {
        mp_stixel_oflow_motion_estimator->compute(m_currRight, m_currLeft, mp_stixel_world_estimator->get_stixels());
    }
    
//     for (uint32_t i = 0; i < mp_stixels_tests.size(); i++) {
//         mp_stixels_tests[i]->set_new_rectified_image(left_view);
//         mp_stixels_tests[i]->updateDenseTracker(m_currLeft);
//         mp_stixels_tests[i]->set_estimated_stixels(mp_stixel_world_estimator->get_stixels());
//         
//         if(mp_video_input->get_current_frame_number() > m_initialFrame)
//             mp_stixels_tests[i]->compute();
//     }
    
//     if (! m_useObjects)
        mp_stixel_motion_evaluator->evaluatePerFrame(mp_video_input->get_current_frame_number() - m_frameBufferLength - 1, m_increment); 
//     else
//         mp_stixel_motion_evaluator->evaluatePerFrameWithObstacles(mp_video_input->get_current_frame_number() - 1);
//     mp_stixel_motion_evaluator->evaluateDisparity(left_view, right_view,
//                                                   mp_video_input->get_current_frame_number() - 1);
    
    cout << "Time for " << __FUNCTION__ << ": " << omp_get_wtime() - startWallTime << endl;
    
    return true;
}

bool StixelsApplicationROS::rectifyPolar()
{
    const double & startWallTime = omp_get_wtime();
    
    if (! m_doPolarCalib)
        return true;
    
    if (m_frameBufferLeft.size() < m_frameBufferLength)
        return false;
    
    cv::Mat prevLeft, prevRight, currRight, FL, FR;
    gil2opencv(boost::gil::view(m_frameBufferLeft[0]), prevLeft);
    gil2opencv(boost::gil::view(m_frameBufferRight[0]), prevRight);

    //     gil2opencv(mp_video_input->get_left_image(), m_currLeft);
    gil2opencv(mp_video_input->get_right_image(), currRight);
    vector < vector < cv::Point2f > > correspondences;
    
    if (! FundamentalMatrixEstimator::findF(prevLeft, prevRight, m_currLeft, currRight, FL, FR, correspondences, 50))
        return false;
    
    if (!  mp_polarCalibration->compute(prevLeft, m_currLeft, FL, correspondences[0], correspondences[3])) {
        cout << "Error while trying to get the polar alignment for the images in the left" << endl;
        return false;
    }
    
    mp_polarCalibration->rectifyAndStoreImages(prevLeft, m_currLeft);
    
    cout << "Time for " << __FUNCTION__ << ": " << omp_get_wtime() - startWallTime << endl;
    
    return true;
}

void StixelsApplicationROS::transformStixels()
{
    const stixels_t & prevStixels = *mp_prevStixels;
    const stixels_t & currStixels = mp_stixel_world_estimator->get_stixels();
    
//     stixels_t m_tStixelsLt0, m_tStixelsLt1;
    vector<cv::Point2d> basePointsLt0(prevStixels.size()), topPointsLt0(prevStixels.size());
    vector<cv::Point2d> basePointsLt1(currStixels.size()), topPointsLt1(currStixels.size());
    
//     vector<cv::Point2d> basePointsTransfLt0, topPointsTransfLt0, basePointsTransfLt1, topPointsTransfLt1;
    
    for (uint32_t i = 0; i < prevStixels.size(); i++) {
        basePointsLt0[i] = cv::Point2d(prevStixels[i].x, prevStixels[i].bottom_y);
        topPointsLt0[i] = cv::Point2d(prevStixels[i].x, prevStixels[i].top_y);
    }

    for (uint32_t i = 0; i < currStixels.size(); i++) {
        basePointsLt1[i] = cv::Point2d(currStixels[i].x, currStixels[i].bottom_y);
        topPointsLt1[i] = cv::Point2d(currStixels[i].x, currStixels[i].top_y);
    }
    
    mp_polarCalibration->transformPoints(basePointsLt0, basePointsTransfLt0, 1);
    mp_polarCalibration->transformPoints(topPointsLt0, topPointsTransfLt0, 1);
//     mp_polarCalibration->transformPoints(basePointsLt1, basePointsTransfLt1, 1);
//     mp_polarCalibration->transformPoints(topPointsLt1, topPointsTransfLt1, 1);
    
    //TODO: Use stixel_t as data type. Then store them into a global variable, so previous transformed stixels
    // are not calculated again. Change this in the visualization part
}

void drawLine(cv::Mat & img, const cv::Point2d & p1, 
              const cv::Point2d & p2, const cv::Scalar & color) {

    if ((p1 != cv::Point2d(0, 0)) && (p2 != cv::Point2d(0, 0))) {
        const cv::Point2d & pA = (p1.y < p2.y)? p1 : p2;
        const cv::Point2d & pB = (p1.y < p2.y)? p2 : p1;
        
        const double dist1 = pA.y + img.rows - pB.y;
        const double dist2 = pB.y - pA.y;
        
        if (dist1 >= dist2) {
            cv::line(img, pA, pB, color);
        } else {
            cv::line(img, pA, cv::Point2d(pA.x, 0), color);
            cv::line(img, pB, cv::Point2d(pA.x, img.rows), color);
        }
    }
}

void StixelsApplicationROS::visualize2()
{
    if (mp_video_input->get_current_frame_number() == m_initialFrame)
        return;
    
    cv::Mat img1Current, img2Current, imgTracking;
    cv::Mat img1Prev, img2Prev;
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), img1Current);
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_right_image()), img2Current);
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), imgTracking);
    
    cv::Mat output = cv::Mat::zeros(600, 1200, CV_8UC3);
    
    cv::Mat scale;
    gil2opencv(boost::gil::view(m_prevLeftRectified), img1Prev);
    gil2opencv(boost::gil::view(m_prevRightRectified), img2Prev);

    cv::Mat Lt0, Lt1;
    mp_polarCalibration->getRectifiedImages(img1Prev, img1Current, Lt0, Lt1);
    
    if (!Lt0.empty() && !Lt1.empty()) {
        if (mp_video_input->get_current_frame_number() != m_initialFrame) {
            for (uint32_t i = 0; i < mp_prevStixels->size(); i++) {
                const cv::Point2d & p1bLin = cv::Point2d(mp_prevStixels->at(i).x, mp_prevStixels->at(i).bottom_y);
                const cv::Point2d & p2bLin = cv::Point2d(mp_stixel_world_estimator->get_stixels().at(i).x, 
                                                     mp_stixel_world_estimator->get_stixels().at(i).bottom_y);
                const cv::Point2d & p1tLin = cv::Point2d(mp_prevStixels->at(i).x, mp_prevStixels->at(i).top_y);
                const cv::Point2d & p2tLin = cv::Point2d(mp_stixel_world_estimator->get_stixels().at(i).x, 
                                                          mp_stixel_world_estimator->get_stixels().at(i).top_y);
                
                const cv::Point2d & p1bPolar = basePointsTransfLt0[i];
                const cv::Point2d & p2bPolar = basePointsTransfLt1[i];
                const cv::Point2d & p1tPolar = topPointsTransfLt0[i];
                const cv::Point2d & p2tPolar = topPointsTransfLt1[i];
                
                cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
                drawLine(Lt0, p1bPolar, p2bPolar, color);
                drawLine(Lt0, p1tPolar, p2tPolar, color);
                
                cv::circle(img1Prev, p1bLin, 2, color, -1);
                cv::circle(img1Prev, p1tLin, 2, color, -1);
                cv::circle(img1Current, p2bLin, 2, color, -1);
                cv::circle(img1Current, p2tLin, 2, color, -1);
                if (mp_stixel_motion_estimator) {
                    const doppia::AbstractStixelMotionEstimator::stixels_motion_t & 
                            corresp = mp_stixel_motion_estimator->get_stixels_motion();
                    int32_t idx = corresp[i];
                    if (idx >= 0) {
                        const cv::Point2d & p1bLin = cv::Point2d(mp_prevStixels->at(idx).x, 
                                                                 mp_prevStixels->at(idx).bottom_y);
                        
                        cv::circle(imgTracking, p1bLin, 2, color, -1);
                        cv::line(imgTracking, p1bLin, p2bLin, color);
                    }
                    cv::circle(imgTracking, p2bLin, 2, color, -1);
                }
            }
        }
        
        cv::resize(Lt0, scale, cv::Size(400, 600));
        scale.copyTo(output(cv::Rect(400, 0, 400, 600)));
    }
    
    
    cv::resize(img1Prev, scale, cv::Size(400, 300));
    scale.copyTo(output(cv::Rect(0, 0, 400, 300)));
    
    cv::resize(img1Current, scale, cv::Size(400, 300));
    scale.copyTo(output(cv::Rect(0, 300, 400, 300)));
    
    cv::resize(imgTracking, scale, cv::Size(400, 300));
    scale.copyTo(output(cv::Rect(800, 0, 400, 300)));
    
    cv::imshow("output", output);
        
    //NOTE: Remove after debugging
//     {
//         gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), img1Current);
//         gil2opencv(boost::gil::view(m_prevLeftRectified), img1Prev);
//         mp_polarCalibration->getRectifiedImages(img1Prev, img1Current, Lt0, Lt1);
//         cv::Mat saveImg(Lt0.rows, 3 * Lt0.cols, CV_8UC3);
//         Lt0.copyTo(saveImg(cv::Rect(0, 0, Lt0.cols, Lt0.rows)));
//         Lt1.copyTo(saveImg(cv::Rect(Lt0.cols, 0, Lt1.cols, Lt1.rows)));
//         saveImg(cv::Rect(Lt0.cols * 2, 0, Lt0.cols, Lt0.rows)) = Lt0 - Lt1;
//         char testName[1024];
//         sprintf(testName, "/tmp/results/img%04d.png", mp_video_input->get_current_frame_number());
//         cv::imwrite(string(testName), saveImg);
//     }
    // end of NOTE
    
    waitForKey(&m_waitTime);
}

void StixelsApplicationROS::visualize3()
{
    const double & startWallTime = omp_get_wtime();
    
    if (mp_video_input->get_current_frame_number() == m_initialFrame)
        return;
    
    cv::Mat imgCurrent, imgPrev;
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), imgCurrent);
    gil2opencv(boost::gil::view(m_prevLeftRectified), imgPrev);
    
    if (mp_stixel_motion_estimator) {
        stixels_t prevStixels = mp_stixel_motion_estimator->get_previous_stixels();
        stixels_t currStixels = mp_stixel_motion_estimator->get_current_stixels();
        AbstractStixelMotionEstimator::stixels_motion_t corresp = mp_stixel_motion_estimator->get_stixels_motion();
        
        for (uint32_t i = 0; i < prevStixels.size(); i++) {
            const cv::Point2d p1a(prevStixels[i].x, prevStixels[i].bottom_y);
            const cv::Point2d p1b(prevStixels[corresp[i]].x, prevStixels[corresp[i]].bottom_y);
            const cv::Point2d p2(currStixels[i].x, currStixels[i].bottom_y);
            
            if (corresp[i] < 0)
                continue;
            
            const cv::Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
            
            cv::circle(imgPrev, p1a, 1, color);
            
            cv::line(imgCurrent, p1b, p2, color);
            
            cv::circle(imgCurrent, p1b, 1, color);
        }
    
        cv::Mat output = cv::Mat::zeros(m_prevLeftRectified.height(), 2 * m_prevLeftRectified.width(), CV_8UC3);
        cv::Mat topView;
        mp_stixel_motion_estimator->drawTracker(imgPrev, topView);
        imgPrev.copyTo(output(cv::Rect(0, 0, imgPrev.cols, imgPrev.rows)));
        topView.copyTo(output(cv::Rect(imgPrev.cols, 0, topView.cols, topView.rows)));
    
//     imgCurrent.copyTo(output(cv::Rect(imgPrev.cols, 0, imgCurrent.cols, imgCurrent.rows)));
    
        cv::imshow("output", output);
    }
    
    cout << "Time for " << __FUNCTION__ << ": " << omp_get_wtime() - startWallTime << endl;
    
    waitForKey(&m_waitTime);
}

void StixelsApplicationROS::visualize()
{
//     if (mp_video_input->get_current_frame_number() == m_initialFrame)
//         return;
    
    if (m_frameBufferLeft.size() < m_frameBufferLength)
        return;
    
    if (mp_polarCalibration) {
        cv::Mat polarOutput;
        cv::Mat polar1, polar2, diffPolar;
        mp_polarCalibration->getStoredRectifiedImages(polar1, polar2);
        cv::Mat inverseX, inverseY;
        mp_polarCalibration->getInverseMaps(inverseX, inverseY, 1);
        cv::absdiff(polar1, polar2, diffPolar);
        cv::Mat diffRect = cv::Mat::zeros(mp_video_input->get_left_image().height(), mp_video_input->get_left_image().width(), CV_8UC3);
        cv::remap(diffPolar, diffRect, inverseX, inverseY, cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
        cv::resize(diffPolar, polarOutput, cv::Size(640, 720));
//         if (mp_stixel_motion_estimator)
//             mp_stixel_motion_estimator->drawTracker(diffRect);
        cv::imshow("polar", diffRect);
    }
    
//     cv::Mat outputDenseTrack;
//     mp_stixel_motion_estimator->drawDenseTracker(outputDenseTrack);
//     cv::imshow("denseTrack", outputDenseTrack);

    if (mp_stixels_tests.size() == 0) {
        visualize3();
        return;
    }
        
    const double & startWallTime = omp_get_wtime();
    
    cv::Mat imgCurrent[6], topView[6];
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), imgCurrent[0]);
    mp_stixels_tests[0]->drawTracker(imgCurrent[0], topView[0]);
    
    cv::Size topSize = cv::Size(imgCurrent[0].cols / 2, imgCurrent[0].rows / 2);
    cv::Mat output = cv::Mat::zeros(2 * imgCurrent[0].rows + topSize.height, 3 * m_prevLeftRectified.width(), CV_8UC3);
    
    cv::Mat topScaled;
    cv::resize(topView[0], topScaled, topSize);
    topScaled.copyTo(output(cv::Rect(0, 2 * imgCurrent[0].rows, topScaled.cols, topScaled.rows)));
    
    for (uint32_t i = 1; i < mp_stixels_tests.size(); i++) {
        imgCurrent[0].copyTo(imgCurrent[i]);
        imgCurrent[i] = cv::Mat(imgCurrent[0].rows, imgCurrent[0].cols, CV_8UC3);
        mp_stixels_tests[i]->drawTracker(imgCurrent[i], topView[i]);
        
        cv::resize(topView[i], topScaled, topSize);
        topScaled.copyTo(output(cv::Rect(topScaled.cols * i, 2 * imgCurrent[0].rows, topScaled.cols, topScaled.rows)));
    }
    
    imgCurrent[0].copyTo(output(cv::Rect(0, 0, imgCurrent[0].cols, imgCurrent[0].rows)));
    imgCurrent[1].copyTo(output(cv::Rect(imgCurrent[0].cols, 0, imgCurrent[1].cols, imgCurrent[1].rows)));
    imgCurrent[2].copyTo(output(cv::Rect(2 * imgCurrent[0].cols, 0, imgCurrent[2].cols, imgCurrent[2].rows)));
    imgCurrent[3].copyTo(output(cv::Rect(0, imgCurrent[0].rows, imgCurrent[3].cols, imgCurrent[3].rows)));
    imgCurrent[4].copyTo(output(cv::Rect(imgCurrent[0].cols, imgCurrent[0].rows, imgCurrent[4].cols, imgCurrent[4].rows)));
    imgCurrent[5].copyTo(output(cv::Rect(2 * imgCurrent[0].cols, imgCurrent[0].rows, imgCurrent[5].cols, imgCurrent[5].rows)));
    
    cv::imshow("output", output);
    
    cout << "Time for " << __FUNCTION__ << ": " << omp_get_wtime() - startWallTime << endl;
    
//     waitForKey(&m_waitTime);
        
}

void StixelsApplicationROS::publishPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & pointCloud) {
    
    cout << "Publishing Stixels" << endl;   
    
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg (*pointCloud, cloudMsg);
    cloudMsg.header.frame_id = CAMERA_FRAME_ID;
    cloudMsg.header.stamp = ros::Time();
         
    m_pointCloudPub.publish(cloudMsg);
         
    ros::spinOnce();
}

void StixelsApplicationROS::publishStixels()
{
    cv::Mat imgLeft;
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), imgLeft);
    
    const stixels_t & stixels = mp_stixel_world_estimator->get_stixels();
//     const stixels3d_t & stixels = mp_stixel_motion_estimator->getLastStixelsAfterTracking();
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (stixels_t::const_iterator it = stixels.begin(); it != stixels.end(); it++) {
        cv::Point2i p1(it->x, it->bottom_y);
        cv::Point2i p2(it->x, it->top_y);

        const doppia::MetricStereoCamera& camera = mp_video_input->get_metric_camera();
        const double & camera_height = mp_video_input->camera_height;
        
        double depth = std::numeric_limits<double>::max();
        if (it->disparity > 0.0f)
            depth = camera.disparity_to_depth(it->disparity );
            
        for (uint32_t y = it->top_y; y <= it->bottom_y; y++) {
            
            Eigen::Vector2f point2d;
            point2d << it->x, y;
            
            
            Eigen::Vector3f point3d = camera.get_left_camera().back_project_2d_point_to_3d(point2d, depth);
            
            cv::Vec3b pixel = imgLeft.at<cv::Vec3b>(y, it->x);
            
            pcl::PointXYZRGB point;
            point.x = point3d(0);
            point.y = point3d(2);
            point.z = camera_height - point3d(1);
            point.r = pixel[2];
            point.g = pixel[1];
            point.b = pixel[0];
            pointCloud->push_back(point);
        }
        
//         cv::line(imgLeft, p1, p2, cv::Scalar(0, 255, 0));
        imgLeft.at<cv::Vec3b>(p1.y, p1.x) = cv::Vec3b(0, 255, 0);
        imgLeft.at<cv::Vec3b>(p2.y, p2.x) = cv::Vec3b(0, 255, 0);
    }
    
    
    // TODO: Get these values from somewhere
//     double deltaTime = 0.1;
//     
//     double posX = 0.0;
//     double posY = 0.0;
//     double posTheta = 0.0; 
//     m_accTime += deltaTime;
//     
//     cout << "accTime " << m_accTime << endl;
// 
//     static tf::TransformBroadcaster broadcaster;
//     tf::StampedTransform transform;
//     // TODO: In a real application, time should be taken from the system
//     transform.stamp_ = ros::Time();
//     transform.setOrigin(tf::Vector3(posX, posY, m_accTime));
//     transform.setRotation( tf::createQuaternionFromRPY(0.0, 0.0, posTheta) );
// 
    publishPointCloud(pointCloud);
//     const tf::StampedTransform stamped = tf::StampedTransform(transform, ros::Time::now(), "/map", "/odom");
//     cout << "stamped " << stamped.stamp_ << endl;
//     broadcaster.sendTransform(stamped);
//     
//     
//     cv::imshow("imgLeft", imgLeft);
//     
//     waitForKey(&m_waitTime);
}

void StixelsApplicationROS::publishStixelsInObjects()
{
    cv::Mat imgLeft;
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), imgLeft);
    
    const StixelsTracker::t_obstaclesTracker & obstaclesTracker = (/*(StixelsTracker)*/mp_stixel_motion_estimator)->getObstaclesTracker();
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    BOOST_FOREACH(const StixelsTracker::t_obstaclesTrack & obstacleTrack, obstaclesTracker) {
        if (obstacleTrack.validCount >= 0) {
            const StixelsTracker::t_obstacle & obstacle = obstacleTrack.track[0];
            
            
            BOOST_FOREACH(Stixel stixel, obstacle.stixels) {
                
//                 stixel.disparity = obstacle.disparity;
                
                cv::Point2i p1(stixel.x, stixel.bottom_y);
                cv::Point2i p2(stixel.x, stixel.top_y);
                
                const doppia::MetricStereoCamera& camera = mp_video_input->get_metric_camera();
                const double & camera_height = mp_video_input->camera_height;
                
                double depth = std::numeric_limits<double>::max();
                if (stixel.disparity > 0.0f)
                    depth = camera.disparity_to_depth(stixel.disparity );
                
                for (uint32_t y = stixel.top_y; y <= stixel.bottom_y; y++) {
                    
                    Eigen::Vector2f point2d;
                    point2d << stixel.x, y;
                    
                    
                    Eigen::Vector3f point3d = camera.get_left_camera().back_project_2d_point_to_3d(point2d, depth);
                    
                    cv::Vec3b pixel = imgLeft.at<cv::Vec3b>(y, stixel.x);
                    
                    pcl::PointXYZRGB point;
                    point.x = point3d(0);
                    point.y = point3d(2);
                    point.z = camera_height - point3d(1);
                    point.r = pixel[2];
                    point.g = pixel[1];
                    point.b = pixel[0];
                    pointCloud->push_back(point);
                }
            }
        }
    }
    
    // TODO: Get these values from somewhere
//     double deltaTime = 0.1;
//     
//     double posX = 0.0;
//     double posY = 0.0;
//     double posTheta = 0.0; 
//     m_accTime += deltaTime;
//     
//     cout << "accTime " << m_accTime << endl;
//     
//     static tf::TransformBroadcaster broadcaster;
//     tf::StampedTransform transform;
//     // TODO: In a real application, time should be taken from the system
//     transform.stamp_ = ros::Time();
//     transform.setOrigin(tf::Vector3(posX, posY, m_accTime));
//     transform.setRotation( tf::createQuaternionFromRPY(0.0, 0.0, posTheta) );
//     
    publishPointCloud(pointCloud);
//     const tf::StampedTransform stamped = tf::StampedTransform(transform, ros::Time::now(), "/map", "/odom");
//     cout << "stamped " << stamped.stamp_ << endl;
//     broadcaster.sendTransform(stamped);
}

void StixelsApplicationROS::publishROS()
{
    const double deltaTime = 1.0/15.0;
    m_accTime += deltaTime;

    rosgraph_msgs::Clock clockMsg;
    clockMsg.clock = ros::Time(m_accTime);
    m_clockPub.publish(clockMsg);
    
//     publishStixels();
    publishStixelsInObjects();
    publishFakePointCloud();
//     publishStereoPointCloud();
    
    sensor_msgs::Image msgLeft, msgRight;
    cv_bridge::CvImage tmpLeft(msgLeft.header, sensor_msgs::image_encodings::BGR8, m_currLeft);
    cv_bridge::CvImage tmpRight(msgRight.header, sensor_msgs::image_encodings::BGR8, m_currRight);
    m_leftCameraInfo.header.stamp = tmpLeft.header.stamp = ros::Time::now();
    m_rightCameraInfo.header.stamp = tmpRight.header.stamp = ros::Time::now();
    
    m_leftInfoPub.publish(m_leftCameraInfo);
    m_righttInfoPub.publish(m_rightCameraInfo);
    
    m_leftImgPub.publish(tmpLeft.toImageMsg());
    m_rightImgPub.publish(tmpRight.toImageMsg());
    

        
    ros::spinOnce();
}


void StixelsApplicationROS::publishFakePointCloud()
{
    const double radius = 15.0;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pointCloud->reserve(4 * radius / 0.01);
    
    for (double x = -radius; x <= radius; x += 0.01) {
        pcl::PointXYZ point;
        point.x = x;
        point.y = -radius;
        point.z = 0.0;
        
        pointCloud->push_back(point);
        
        point.y = radius;
        pointCloud->push_back(point);
    }
    
    for (double y = -radius; y <= radius; y += 0.01) {
        pcl::PointXYZ point;
        point.x = -radius;
        point.y = y;
        point.z = 0.0;
        
        pointCloud->push_back(point);
        
        point.x = radius;
        pointCloud->push_back(point);
    }
    
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg (*pointCloud, cloudMsg);
    cloudMsg.header.frame_id = CAMERA_FRAME_ID;
    cloudMsg.header.stamp = ros::Time::now();
    cloudMsg.header.seq = mp_video_input->get_current_frame_number();
    
    m_fakePointCloudPub.publish(cloudMsg);
}

void StixelsApplicationROS::publishStereoPointCloud()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pointCloud->reserve(m_currLeft.rows * m_currLeft.cols);
    cv::Mat leftGray, rightGray;
    cv::Mat dispELAS, scaledMapELAS, colorMapELAS, maskELAS;
    cv::cvtColor(m_currRight, leftGray, CV_BGR2GRAY);
    cv::cvtColor(m_currLeft, rightGray, CV_BGR2GRAY);
    
    Elas elas(Elas::parameters(Elas::ROBOTICS));
    
    const doppia::MetricStereoCamera& camera = mp_video_input->get_metric_camera();
    const double & camera_height = mp_video_input->camera_height;
    
    dispELAS = cv::Mat(m_currLeft.rows, m_currLeft.cols, CV_64F);
    
    int32_t dims[3];
    dims[0] = leftGray.cols;
    dims[1] = leftGray.rows;
    dims[2] = leftGray.cols;
    
    float * D1 = new float[leftGray.cols * leftGray.rows];
    float * D2 = new float[rightGray.cols * rightGray.rows];
    
    elas.process((uint8_t *)leftGray.data, (uint8_t *)rightGray.data, D1, D2, dims);
    
    for (uint32_t y = 0; y < leftGray.rows; y++) {
        for (uint32_t x = 0; x < leftGray.cols; x++) {
            dispELAS.at<double>(y, x) = D1[y * leftGray.cols + x];
            
            Eigen::Vector2f point2d;
            point2d << y, x;
            const Eigen::Vector3f & point3d = camera.get_left_camera().back_project_2d_point_to_3d(point2d, D1[y * leftGray.cols + x] / 5.0);
            
            cv::Vec3b pixel = m_currRight.at<cv::Vec3b>(y, x);
            
            pcl::PointXYZRGB point;
            point.z = camera_height - point3d(0);
            point.y = point3d(2);
            point.x = point3d(1);
            point.r = pixel[2];
            point.g = pixel[1];
            point.b = pixel[0];
            pointCloud->push_back(point);
        }
    }
    
    delete D1;
    delete D2;
    
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg (*pointCloud, cloudMsg);
    cloudMsg.header.frame_id = CAMERA_FRAME_ID;
    cloudMsg.header.stamp = ros::Time();
    
    m_stereoPointCloudPub.publish(cloudMsg);
    
    ros::spinOnce();

//     const cv::Mat & map, cv::Mat & falseColorsMap, cv::Mat & scaledMap
//     dispELAS, colorMapELAS, scaledMapELAS
    double min;
    double max;
    cv::minMaxIdx(dispELAS, &min, &max);
    max = 64;
    min = 0;
    // expand your range to 0..255. Similar to histEq();
    dispELAS.convertTo(scaledMapELAS, CV_8UC1, 255 / (max-min), min); 
    
    applyColorMap(dispELAS, colorMapELAS, cv::COLORMAP_JET);    
    
    cv::imshow("colorMapELAS", colorMapELAS);
    cv::waitKey(20);
}

}