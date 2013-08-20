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
    m_options = parseOptionsFile(optionsFile);
    
    mp_video_input.reset(doppia::VideoInputFactory::new_instance(m_options));
    
    if(not mp_video_input)
    {
        throw std::invalid_argument("Failed to initialize a video input module. "
        "No images to read, nothing to compute.");
    }
    
    mp_stixel_world_estimator.reset(doppia::StixelWorldEstimatorFactory::new_instance(m_options, *mp_video_input));
    
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
    // Updating the rectified images
    stixel_world::input_image_const_view_t currLeft = mp_video_input->get_left_image();
    stixel_world::input_image_const_view_t currRight = mp_video_input->get_right_image();
    
    boost::gil::copy_pixels(currLeft, boost::gil::view(m_prevLeftRectified));
    boost::gil::copy_pixels(currRight, boost::gil::view(m_prevRightRectified));
}

bool StixelsApplication::iterate()
{
    if (! mp_video_input->next_frame())
        return false;
        
    if (mp_linearRectification.use_count() == 0) {
        boost::program_options::variables_map options = m_options;
        modify_variable_map(options, "preprocess.rectify", true);

        mp_linearRectification.reset(new doppia::CpuPreprocessor(
            mp_video_input->get_left_image().dimensions(), mp_video_input->get_stereo_calibration(), options));

//         mp_linearRectification.reset((doppia::CpuPreprocessor *)doppia::VideoInputFactory::new_instance(options));
        
        m_prevLeftRectified = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_left_image().dimensions());
        m_prevRightRectified = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_right_image().dimensions());
        
        m_currentLeft = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_left_image().dimensions());
        m_currentRight = doppia::AbstractVideoInput::input_image_t(mp_video_input->get_right_image().dimensions());
    }
    
//     mp_linearRectification->run(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), 0, boost::gil::view(m_currentLeft));
//     mp_linearRectification->run(stixel_world::input_image_const_view_t(mp_video_input->get_right_image()), 1, boost::gil::view(m_currentRight));
        mp_linearRectification->run(stixel_world::input_image_const_view_t(mp_video_input->get_left_image()), 0, 
                                    doppia::CpuPreprocessor::output_image_view_t(m_currentLeft._view));
        mp_linearRectification->run(stixel_world::input_image_const_view_t(mp_video_input->get_right_image()), 1, 
                                    doppia::CpuPreprocessor::output_image_view_t(m_currentRight._view));

        doppia::AbstractStixelWorldEstimator::input_image_const_view_t
            left_view(m_currentLeft._view),
            right_view(m_currentRight._view);        

        //         left_view(mp_video_input->get_left_image()),
//         right_view(mp_video_input->get_right_image());
    
    mp_stixel_world_estimator->set_rectified_images_pair(left_view, right_view);
    mp_stixel_world_estimator->compute();
    
    m_currStixels = mp_stixel_world_estimator->get_stixels();
    
    return true;
}

void StixelsApplication::visualize()
{
    cv::Mat img1Current, img2Current;
    cv::Mat img1Prev, img2Prev;
    gil2opencv(stixel_world::input_image_const_view_t(boost::gil::view(m_currentLeft)), img1Current);
    gil2opencv(stixel_world::input_image_const_view_t(boost::gil::view(m_currentRight)), img2Current);
    
    for (uint32_t i = 0; i < m_currStixels.size(); i++) {
        img1Current.at<cv::Vec3b>(m_currStixels[i].bottom_y, i) = cv::Vec3b(0, 0, 255);
        img1Current.at<cv::Vec3b>(m_currStixels[i].top_y, i) = cv::Vec3b(255, 0, 0);
    }
    
    cv::imshow("img1", img1Current);
    cv::imshow("img2", img2Current);
    
    if (mp_video_input->get_current_frame_number() != 1) {
        gil2opencv(boost::gil::view(m_prevLeftRectified), img1Prev);
        gil2opencv(boost::gil::view(m_prevRightRectified), img2Prev);

        cv::imshow("img1Prev", img1Prev);
        cv::imshow("img2Prev", img2Prev);
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



