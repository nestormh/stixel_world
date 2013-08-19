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

#include <boost/filesystem.hpp>
#include <fstream>

using namespace stixel_world;

StixelsApplication::StixelsApplication(const string& optionsFile)
{
    boost::program_options::variables_map options = parseOptionsFile(optionsFile);
    
    m_video_input_p.reset(doppia::VideoInputFactory::new_instance(options));
    
    if(not m_video_input_p)
    {
        throw std::invalid_argument("Failed to initialize a video input module. "
        "No images to read, nothing to compute.");
    }
    
    m_stixel_world_estimator_p.reset(doppia::StixelWorldEstimatorFactory::new_instance(options, *m_video_input_p));
    
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
    cv::namedWindow("img1");
    cv::namedWindow("img2");
    cv::moveWindow("img2", 700, 0);
    
    while (iterate()) {
        visualize();
        update();
    }
    
}

void StixelsApplication::update()
{
    stixel_world::input_image_const_view_t currLeft = m_video_input_p->get_left_image();
    m_prevLeftRectified = doppia::AbstractVideoInput::input_image_t(currLeft.dimensions());
    boost::gil::copy_pixels(currLeft, boost::gil::view(m_prevLeftRectified));
    
    cout << m_prevLeftRectified.dimensions().x << endl;
}


bool StixelsApplication::iterate()
{
    if (! m_video_input_p->next_frame())
        return false;
    
    stixel_world::input_image_const_view_t
        left_view(m_video_input_p->get_left_image()),
        right_view(m_video_input_p->get_right_image());
    
    m_stixel_world_estimator_p->set_rectified_images_pair(left_view, right_view);
    m_stixel_world_estimator_p->compute();
    
//     stixels_t stixels = m_stixel_world_estimator_p->get_stixels();
//     m_currStixels.reset(&stixels);
    
    return true;
}

void StixelsApplication::visualize()
{
    cv::Mat img1Current, img2Current;
    cv::Mat img1Prev, img2Prev;
    gil2opencv(stixel_world::input_image_const_view_t(m_video_input_p->get_left_image()), img1Current);
    gil2opencv(stixel_world::input_image_const_view_t(m_video_input_p->get_right_image()), img2Current);
    
//     for (uint32_t i = 0; i < m_currStixels->size(); i++) {
//         img1Current.at<cv::Vec3b>((*m_currStixels)[i].bottom_y, i) = cv::Vec3b(0, 0, 255);
//         img1Current.at<cv::Vec3b>((*m_currStixels)[i].top_y, i) = cv::Vec3b(255, 0, 0);
//     }
    
    cv::imshow("img1", img1Current);
    cv::imshow("img2", img2Current);
    
    if (m_video_input_p->get_current_frame_number() != 1) {
        cout << "viz " << m_prevLeftRectified.dimensions().x << endl;
//         doppia::AbstractVideoInput::input_image_t::view_t viewLeft(m_prevLeftRectified);
        gil2opencv(boost::gil::view(m_prevLeftRectified), img1Prev);
//         gil2opencv(m_prevRightRectified, img2Prev);
//         
        cv::imshow("img1Prev", img1Prev);
//         cv::imshow("img2Prev", img2Prev);
    }
    
    uint8_t keycode = cv::waitKey(0);
    switch (keycode) {
        case 'q':
            exit(0);
            break;
        default:
            ;
    }
}



