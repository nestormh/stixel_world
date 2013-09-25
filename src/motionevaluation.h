/*
    Copyright 2013 Néstor Morales Hernández <email>

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


#ifndef MOTIONEVALUATION_H
#define MOTIONEVALUATION_H

#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>

#include "stixelstracker.h"

#include "stereo_matching/stixels/Stixel.hpp"
#include "stereo_matching/stixels/stixels.pb.h"
#include "stereo_matching/stixels/AbstractStixelWorldEstimator.hpp"
#include "helpers/data/DataSequence.hpp"

#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

namespace stixel_world {
    
#define MAX_LENGTH 51
    
typedef struct {
    cv::Point2i ul;
    cv::Point2i br;
    int32_t score;
    uint32_t objectId;
} t_annotation;

typedef struct {
    uint32_t tp;
    uint32_t fn;
    double recallSum;
    uint32_t totalExamples;
} t_statistics_counter;

typedef DataSequence<doppia_protobuf::Stixels> StixelsDataSequence;

typedef struct {
    boost::shared_ptr<doppia::AbstractStixelWorldEstimator> p_stixel_world_estimator;
    boost::shared_ptr<StixelsTracker> p_stixel_motion_estimator;
    vector <t_statistics_counter> counters;
    boost::shared_ptr<StixelsDataSequence> stixels_data_sequence;
} t_statistics_handler;

class MotionEvaluation
{
public:
    MotionEvaluation(const boost::program_options::variables_map &options);
    ~MotionEvaluation();
    static boost::program_options::options_description get_args_options();
    
    void addStixelMotionEstimator(const boost::shared_ptr<doppia::AbstractStixelWorldEstimator> & p_stixel_world_estimator, 
                                  const boost::shared_ptr<StixelsTracker> & p_stixel_motion_estimator);
    
    void evaluate(const uint32_t & currentFrame);

protected:
    void parseArguments(const boost::program_options::variables_map &options);
    void readAnnotationFile(const string & annotationFileName);
    
    bool isTheSameAnnotation(const t_annotation& newAnnotation, const t_annotation& oldAnnotation);
    
    int32_t getStixelObjectId(const doppia::Stixel & stixel, const vector<t_annotation> & annotations);
    
    bool m_evaluationActivated;
    boost::filesystem::path m_outputFolder;
    boost::filesystem::path m_outputFileName;
    
    vector < vector<t_annotation> > m_annotations;
    
    vector < t_statistics_handler > m_statistics_handlers;
    
    float area(const t_annotation & a);
    float overlappingArea(const t_annotation & a, const t_annotation & b);
    bool intersectionOverUnionCriterion(const t_annotation & detection, const t_annotation & gt, const float & p);
    bool doOverlap(const t_annotation & a, const t_annotation & b);
    void countErrors(const float& detectionThresh, const vector< t_annotation >& gt, const vector< t_annotation >& detections,
                     uint32_t& tp, uint32_t& fn);
    
    void getAnnotationsFromTracks(const StixelsTracker::t_historic & historic, const uint32_t& idx, const uint32_t & currentFrame,
                                  vector< t_annotation >& detections);
    
    void saveResults();
};


}

#endif // MOTIONEVALUATION_H
