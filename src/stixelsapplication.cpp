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


#include "stixelsapplication.h"

#include "stereo_matching/stixels/StixelWorldEstimatorFactory.hpp"

#include "utils.h"
#include "fundamentalmatrixestimator.h"

#include <boost/filesystem.hpp>
#include <fstream>

using namespace stixel_world;

StixelsApplication::StixelsApplication(const string& optionsFile)
{
    m_options = parseOptionsFile(optionsFile);
    
    mp_video_input.reset(doppia::VideoInputFactory::new_instance(m_options));
    
    if(not mp_video_input)
    {
        throw std::invalid_argument("Failed to initialize a video input module. "
        "No images to read, nothing to compute.");
    }
    
    mp_stixel_world_estimator.reset(doppia::StixelWorldEstimatorFactory::new_instance(m_options, *mp_video_input));
    mp_prevStixels.reset(new stixels_t);
    mp_polarCalibration.reset(new PolarCalibration());
    
    return;
}

boost::program_options::variables_map StixelsApplication::parseOptionsFile(const string& optionsFile) 
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

        init_stixel_world(configurationFilePath);
        boost::program_options::options_description desc;
        get_options_description(desc);
        
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
            cout << "\033[1;31mError parsing THE configuration file named:\033[0m "
            << configurationFilePath << endl;
            cout << desc << endl;
            throw;
        }

        cout << "Parsed the configuration file " << configurationFilePath << std::endl;
    }
    return options;   
}

void StixelsApplication::runStixelsApplication()
{
    cv::namedWindow("linear");
    cv::namedWindow("polar1");
    cv::namedWindow("polar2");
//     cv::namedWindow("img2Prev");
//     cv::namedWindow("img1");
//     cv::namedWindow("img2");
//     cv::namedWindow("Lt0");
//     cv::namedWindow("Rt0");
//     cv::namedWindow("Lt1");
//     cv::namedWindow("Rt1");
//     cv::moveWindow("img2", 700, 0);
//     cv::moveWindow("img1Prev", 0, 400);
//     cv::moveWindow("img2Prev", 700, 400);
//     cv::moveWindow("Lt0", 1, 1);
//     cv::moveWindow("Rt0", 300, 0);
//     cv::moveWindow("Lt1", 600, 0);
//     cv::moveWindow("Rt1", 900, 0);
    
    m_prevLeftRectified = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_left_image().dimensions());
    m_prevRightRectified = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_right_image().dimensions());
    
    while (iterate()) {
        visualize();
        update();
    }
}

void StixelsApplication::update()
{

    // Updating the rectified images
    const stixel_world::input_image_const_view_t & currLeft = mp_video_input->get_left_image();
    const stixel_world::input_image_const_view_t & currRight = mp_video_input->get_right_image();
    
    boost::gil::copy_pixels(currLeft, boost::gil::view(m_prevLeftRectified));
    boost::gil::copy_pixels(currRight, boost::gil::view(m_prevRightRectified));
    
    // Updating the stixels
    mp_prevStixels->resize(mp_stixel_world_estimator->get_stixels().size());
    std::copy(mp_stixel_world_estimator->get_stixels().begin(), 
                mp_stixel_world_estimator->get_stixels().end(), 
                mp_prevStixels->begin());
}

bool StixelsApplication::iterate()
{
    if (! mp_video_input->next_frame())
        return false;
                
    doppia::AbstractVideoInput::input_image_view_t
                        left_view(mp_video_input->get_left_image()),
                        right_view(mp_video_input->get_right_image());    
            
    mp_stixel_world_estimator->set_rectified_images_pair(left_view, right_view);
    mp_stixel_world_estimator->compute();
    
    if (! rectifyPolar()) {
        // TODO: Do something in this case
        return true;
    }
        
    return true;
}

bool StixelsApplication::rectifyPolar()
{
    if (mp_video_input->get_current_frame_number() == 1)
        return true;
    
    cv::Mat prevLeft, prevRight, currLeft, currRight, FL, FR;
    gil2opencv(boost::gil::view(m_prevLeftRectified), prevLeft);
    gil2opencv(boost::gil::view(m_prevRightRectified), prevRight);
    gil2opencv(mp_video_input->get_left_image(), currLeft);
    gil2opencv(mp_video_input->get_right_image(), currRight);
    vector < vector < cv::Point2f > > correspondences;
    
    if (! FundamentalMatrixEstimator::findF(prevLeft, prevRight, currLeft, currRight, FL, FR, correspondences, 10))
        return false;
    
    if (!  mp_polarCalibration->compute(prevLeft, currLeft, FL, correspondences[0], correspondences[3])) {
        cout << "Error while trying to get the polar alignment for the images in the left" << endl;
        return false;
    }
    
    cv::Mat Lt0, Rt0, Lt1, Rt1;
    
    mp_polarCalibration->getRectifiedImages(prevLeft, currLeft, Lt0, Lt1);
    mp_polarCalibration->getRectifiedImages(prevLeft, currLeft, Rt0, Rt1);
    
    m_polarLt0 = doppia::AbstractVideoInput::input_image_t(Lt0.cols, Lt0.rows);
    m_polarRt0 = doppia::AbstractVideoInput::input_image_t(Rt0.cols, Rt0.rows);
    m_polarLt1 = doppia::AbstractVideoInput::input_image_t(Lt1.cols, Lt1.rows);
    m_polarRt1 = doppia::AbstractVideoInput::input_image_t(Rt1.cols, Rt1.rows);
    
    boost::gil::rgb8_view_t viewLt0 = boost::gil::view(m_polarLt0);
    boost::gil::rgb8_view_t viewRt0 = boost::gil::view(m_polarRt0);
    boost::gil::rgb8_view_t viewLt1 = boost::gil::view(m_polarLt1);
    boost::gil::rgb8_view_t viewRt1 = boost::gil::view(m_polarRt1);
    
    opencv2gil(Lt0, viewLt0);
    opencv2gil(Rt0, viewRt0);
    opencv2gil(Lt1, viewLt1);
    opencv2gil(Rt1, viewRt1);
        
    return true;
}

void StixelsApplication::visualize()
{
    if (mp_video_input->get_current_frame_number() == 1)
        return;
    
    cv::Mat img1Current, img2Current;
    cv::Mat img1Prev, img2Prev;
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), img1Current);
    gil2opencv(stixel_world::input_image_const_view_t(mp_video_input->get_right_image()), img2Current);
    
    cv::Mat linearOutput = cv::Mat::zeros(600, 800, CV_8UC3);
    cv::Mat scale;
    cv::resize(img1Current, scale, cv::Size(400, 300));
    scale.copyTo(linearOutput(cv::Rect(0, 300, 400, 300)));
    
    for (uint32_t i = 0; i < mp_stixel_world_estimator->get_stixels().size(); i++) {
//         img1Current.at<cv::Vec3b>(mp_stixel_world_estimator->get_stixels().at(i).bottom_y, i) = cv::Vec3b(0, 0, 255);
//         img1Current.at<cv::Vec3b>(mp_stixel_world_estimator->get_stixels().at(i).top_y, i) = cv::Vec3b(255, 0, 0);
        const cv::Point2d p1(i * mp_stixel_world_estimator->get_stixels().at(i).width, mp_stixel_world_estimator->get_stixels().at(i).bottom_y);
        const cv::Point2d p2((i + 1) * mp_stixel_world_estimator->get_stixels().at(i).width, mp_stixel_world_estimator->get_stixels().at(i).top_y);
        
        cv::rectangle(img1Current, p1, p2, cv::Scalar(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF), -1);
        
        img1Current.at<cv::Vec3b>(img1Current.rows - 128 + mp_stixel_world_estimator->get_stixels().at(i).disparity, i) = cv::Vec3b(0, 0, 255);
    }
    
    if (mp_video_input->get_current_frame_number() != 1) {
        gil2opencv(boost::gil::view(m_prevLeftRectified), img1Prev);
        gil2opencv(boost::gil::view(m_prevRightRectified), img2Prev);
        
        cv::resize(img1Prev, scale, cv::Size(400, 300));
        scale.copyTo(linearOutput(cv::Rect(0, 0, 400, 300)));
        
        for (uint32_t i = 0; i < mp_prevStixels->size(); i++) {
//             img1Prev.at<cv::Vec3b>(mp_prevStixels->at(i).bottom_y, i) = cv::Vec3b(0, 0, 255);
//             img1Prev.at<cv::Vec3b>(mp_prevStixels->at(i).top_y, i) = cv::Vec3b(255, 0, 0);
            
            const cv::Point2d p1(i * mp_prevStixels->at(i).width, mp_prevStixels->at(i).bottom_y);
            const cv::Point2d p2(i * mp_prevStixels->at(i).width, mp_prevStixels->at(i).top_y);
            
            cv::rectangle(img1Prev, p1, p2, cv::Scalar(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF), -1);
            
            img1Prev.at<cv::Vec3b>(img1Prev.rows - 128 + mp_prevStixels->at(i).disparity, i) = cv::Vec3b(0, 0, 255);
        }
        cv::resize(img1Prev, scale, cv::Size(400, 300));
        scale.copyTo(linearOutput(cv::Rect(400, 0, 400, 300)));
    }
    cv::resize(img1Current, scale, cv::Size(400, 300));
    scale.copyTo(linearOutput(cv::Rect(400, 300, 400, 300)));
    
    cv::imshow("linear", linearOutput);
    
    cv::Mat polarOutput1 = cv::Mat::zeros(600, 400, CV_8UC3);
    cv::Mat polarOutput2 = cv::Mat::zeros(600, 400, CV_8UC3);
    
    cv::Mat Lt0, Lt1;
    mp_polarCalibration->getRectifiedImages(img1Prev, img1Current, Lt0, Lt1);
    
    if (!Lt0.empty() && !Lt1.empty()) {
        cv::resize(Lt0, scale, cv::Size(400, 600));
        scale.copyTo(polarOutput1(cv::Rect(0, 0, 400, 600)));
        cv::resize(Lt1, scale, cv::Size(400, 600));
        scale.copyTo(polarOutput2(cv::Rect(0, 0, 400, 600)));
    }
    
    cv::imshow("polar1", polarOutput1);
    cv::imshow("polar2", polarOutput2);
    
//     char testName[1024];
//     sprintf(testName, "/tmp/results/img%04d.png", mp_video_input->get_current_frame_number());
//     cv::imwrite(string(testName), polarOutput1);
    
    waitForKey();
}

