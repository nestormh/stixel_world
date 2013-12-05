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


#include "motionevaluation.h"
#include "helpers/get_option_value.hpp"

#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/graph/graph_concepts.hpp>

#include <helpers/xyz_indices.hpp>

#include "stixelstracker.h"
#include <../../GPUCPD/src/LU-Decomposition/Libs/Cuda/include/CL/cl_platform.h>
#include <../../GPUCPD/src/LU-Decomposition/Libs/Cuda/include/device_launch_parameters.h>

namespace stixel_world {
    
MotionEvaluation::MotionEvaluation(const boost::program_options::variables_map& options)
{
    parseArguments(options);
    
    using namespace boost::posix_time;
    const ptime current_time(second_clock::local_time());
    m_outputFileName = m_outputFolder / 
                            boost::str( boost::format("results_%i_%02i_%02i___%02i_%02i_%02i.txt")
                                            % current_time.date().year()
                                            % current_time.date().month().as_number()
                                            % current_time.date().day()
                                            % current_time.time_of_day().hours()
                                            % current_time.time_of_day().minutes()
                                            % current_time.time_of_day().seconds() );
                            
    BOOST_FOREACH(t_statistics_handler & handler, m_statistics_handlers) {
        for (uint32_t i = 0; i < MAX_LENGTH; i++) {
            handler.counters[i].tp = 0;
            handler.counters[i].fn = 0;
            handler.counters[i].recallSum = 0;
            handler.counters[i].totalExamples = 0;
        }
    }
}

MotionEvaluation::~MotionEvaluation()
{
    saveResults();
}

void MotionEvaluation::saveResults()
{

    using namespace boost::posix_time;
    const ptime current_time(second_clock::local_time());
    ofstream fout(m_outputFileName.c_str(), ios::trunc);
    fout << boost::str( boost::format("%i/%02i/%02i %i:%i:%i\n")
                            % current_time.date().year()
                            % current_time.date().month().as_number()
                            % current_time.date().day()
                            % current_time.time_of_day().hours()
                            % current_time.time_of_day().minutes()
                            % current_time.time_of_day().seconds() );
    {
        uint32_t i = 0;
        BOOST_FOREACH(const t_statistics_handler & handler, m_statistics_handlers) {
            fout << "EXAMPLE " << i << ": ";
            fout << "SAD FACTOR: " << handler.p_stixel_motion_estimator->getSADFactor() << " ||| ";
            fout << "HEIGHT FACTOR: " << handler.p_stixel_motion_estimator->getHeightFactor() << " ||| ";
            fout << "POLAR DIST FACTOR: " << handler.p_stixel_motion_estimator->getPolarDistFactor() << " ||| ";
            fout << "POLAR SAD FACTOR: " << handler.p_stixel_motion_estimator->getPolarSADFactor() << " ||| ";
            fout << "DENSE TRACKING FACTOR: " << handler.p_stixel_motion_estimator->getDenseTrackingFactor() << " ||| ";
            fout << "USES GRAPH: " << handler.p_stixel_motion_estimator->useGraphs() << endl;
            ;
            i++;
        }
    }
    
    for (uint32_t i = 0; i < m_statistics_handlers[0].counters.size(); i++) {
        BOOST_FOREACH(const t_statistics_handler & handler, m_statistics_handlers) {
            fout << (handler.counters[i].tp / (float)(handler.counters[i].tp + handler.counters[i].fn)) << "\t";
        }
        fout << endl;
    }
    fout.close();
}


boost::program_options::options_description MotionEvaluation::get_args_options()
{
    boost::program_options::options_description desc("MotionEvaluation options");
    
    desc.add_options()
    
    ("stixel_world.motion.evaluation.annotations",
     boost::program_options::value<string>()->default_value( "" ),
     "IDL file containing the annotations.")
    
    ("stixel_world.motion.evaluation.output_folder",
     boost::program_options::value<string>()->default_value( "" ),
     "Output folder in which results will be stored.")
    ;
    
    return desc;
}

bool MotionEvaluation::isTheSameAnnotation(const t_annotation& newAnnotation, const t_annotation& oldAnnotation)
{
    cv::Point2i ul(max(newAnnotation.ul.x, oldAnnotation.ul.x), max(newAnnotation.ul.y, oldAnnotation.ul.y));
    cv::Point2i br(min(newAnnotation.br.x, oldAnnotation.br.x), min(newAnnotation.br.y, oldAnnotation.br.y));
    
    const double commonArea = (br.x - ul.x) * (br.y - ul.y);
    const double newArea = (newAnnotation.br.x - newAnnotation.ul.x) * (newAnnotation.br.y - newAnnotation.ul.y);
    
    if ((commonArea / newArea) > 0.75)
        return true;
    else
        return false;
}


void MotionEvaluation::readAnnotationFile(const string& annotationFileName)
{
    try {
        ifstream file(annotationFileName.c_str(), ios_base::in | ios_base::binary);
        
        boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
        in.push(boost::iostreams::gzip_decompressor());
        in.push(file);

        stringstream ss;
        boost::iostreams::copy(in, ss);
        
        m_annotations.clear();
        
        uint32_t lastObstacleId = 0;
        
        while (! ss.eof()) {
            string imageName;
            ss >> imageName;
            
            char delim;
            vector < t_annotation > frameAnnotations;
            while (true) {
                t_annotation annotation;
                
                string ulx, uly, brx, bry, score;
                ss >> ulx;
                ss >> uly;
                ss >> brx;
                ss >> bry;
                
                ulx.replace(ulx.find("("), 1, "");
                ulx.replace(ulx.find(","), 1, "");
                uly.replace(uly.find(","), 1, "");
                brx.replace(brx.find(","), 1, "");
                score = bry.substr(bry.find(":"));
                bry = bry.substr(0, bry.find(")"));
                
                score.replace(score.find(":"), 1, "");
                delim = score[score.size() - 1];
                score.replace(score.find(delim), 1, "");
                                
                int iulx, iuly, ibrx, ibry;
                iulx = atoi(ulx.c_str());
                iuly = atoi(uly.c_str());
                ibrx = atoi(brx.c_str());
                ibry = atoi(bry.c_str());
                
                annotation.ul = cv::Point2i(min(iulx, ibrx), min(iuly, ibry));
                annotation.br = cv::Point2i(max(iulx, ibrx), max(iuly, ibry));  
                annotation.score = atoi(score.c_str());
                
                if (annotation.br.y - annotation.ul.y >= 40)
                    frameAnnotations.push_back(annotation);
                
                if ((delim == ';') ||(delim == '.'))
                    break;
            }
            if (delim == '.')
                break;
            
            for (uint32_t i = 0; i < frameAnnotations.size(); i++) {
                bool objectFound = false;
                if (m_annotations.size() != 0) {
                    const vector < t_annotation > & lastAnnotations = m_annotations[m_annotations.size() - 1];
                    for (uint32_t j = 0; j < lastAnnotations.size(); j++) {
                        if (isTheSameAnnotation(frameAnnotations[i], lastAnnotations[j])) {
                            frameAnnotations[i].objectId = lastAnnotations[j].objectId;
                            objectFound = true;
                            break;
                        }
                    }
                }
                if (! objectFound) {
                    frameAnnotations[i].objectId = lastObstacleId;
                    lastObstacleId++;
                }
            }
            
            m_annotations.push_back(frameAnnotations);
        }
        
//         for (uint32_t i = 0; i < m_annotations.size(); i++) {
//             for (uint32_t j = 0; j < m_annotations[i].size(); j++) {
//                 cout << "( " << m_annotations[i][j].ul << "; " << m_annotations[i][j].br << "):" << m_annotations[i][j].objectId << " || ";
//             }
//             cout << endl;
//         }
        
        file.close();
    } catch(const boost::iostreams::gzip_error& e) {
        std::cout << e.what() << '\n';
    }
}

void MotionEvaluation::parseArguments(const boost::program_options::variables_map& options)
{
    
    m_evaluationActivated = true;
    
    string annotationsFileName;
    if(options.count("stixel_world.motion.evaluation.annotations") > 0)
    {
        annotationsFileName = get_option_value<std::string>(options, "stixel_world.motion.evaluation.annotations");
        
        if (annotationsFileName.size() == 0) {
            cout << "No annotations file provided. No evaluations will be performed" << std::endl;
            m_evaluationActivated = false;
            return;
        }
    }
    else
    {
        cout << "No annotations file provided. No evaluations will be performed" << std::endl;
        m_evaluationActivated = false;
        return;
    }

    readAnnotationFile(annotationsFileName);
    
    if(options.count("stixel_world.motion.evaluation.output_folder") > 0)
    {
        m_outputFolder = boost::filesystem::path(
                                get_option_value<std::string>(options, "stixel_world.motion.evaluation.output_folder"));
    }
    else
    {
        cout << "No output folder provided. Temporary folder will be used" << std::endl;
        m_outputFolder = boost::filesystem::path("/tmp");
        return;
    }
}

void MotionEvaluation::addStixelMotionEstimator(const boost::shared_ptr<doppia::AbstractStixelWorldEstimator> & p_stixel_world_estimator, 
                                                const boost::shared_ptr<StixelsTracker>& p_stixel_motion_estimator)
{
    t_statistics_handler statistics_handler;
    statistics_handler.p_stixel_world_estimator = p_stixel_world_estimator;
    statistics_handler.p_stixel_motion_estimator = p_stixel_motion_estimator;
    statistics_handler.counters = vector<t_statistics_counter>(MAX_LENGTH);
    for (uint32_t i = 0; i < MAX_LENGTH; i++) {
        statistics_handler.counters[i].tp = 0;
        statistics_handler.counters[i].fn = 0;
    }
    
    m_statistics_handlers.push_back(statistics_handler);
}

inline
int32_t MotionEvaluation::getStixelObjectId(const doppia::Stixel& stixel, const vector< t_annotation >& annotations)
{
    for (uint32_t i = 0; i < annotations.size(); i++) {
        if ((annotations[i].ul.x <= stixel.x) && (annotations[i].br.x >= stixel.x)) {
            return annotations[i].objectId;
        }
    }
    return -1;
}

//TODO Check this!
void MotionEvaluation::evaluate(const uint32_t & currentFrame)
{
    
    if ((currentFrame <= MAX_LENGTH) || (! m_evaluationActivated))
        return;
    
    BOOST_FOREACH(t_statistics_handler & handler, m_statistics_handlers) {
        const StixelsTracker::t_historic & historic = (handler.p_stixel_motion_estimator)->getHistoric();
        for (uint32_t j = 1; j <= MAX_LENGTH; j++) {
            const uint32_t evaluatedFrame = currentFrame - j;
            
            vector< t_annotation > detections;
            getAnnotationsFromTracks(historic, j, currentFrame, detections);
                    
            vector < t_annotation> annotations = m_annotations[currentFrame];
            uint32_t tmpTp, tmpFn;
            countErrors(0.0f, annotations, detections, tmpTp, tmpFn);

            handler.counters[j - 1].tp += tmpTp;
            handler.counters[j - 1].fn += tmpFn;
        }
    }
    
    if (currentFrame % 10 == 0)
        saveResults();
}

inline
float MotionEvaluation::area(const t_annotation& a)
{
    const float w = a.br.x - a.ul.x;
    const float h = a.br.y - a.ul.y;
    
    return w * h;
}

inline
float MotionEvaluation::overlappingArea(const t_annotation& a, const t_annotation& b)
{
    const float w = min(a.br.x, b.br.x) - max(a.ul.x, b.ul.x);
    const float h = min(a.br.y, b.br.y) - max(a.ul.y, b.ul.y);    
    
    if ((w < 0) or (h < 0)) {
        return 0.0f;
    } else {
        return float(w * h);
    }
}

inline
bool MotionEvaluation::doOverlap(const t_annotation& d, const t_annotation& gt)
{
//     return ((a.ul.x <= b.br.x) &&
//             (a.br.x >= b.ul.x) &&
//             (a.ul.y <= b.br.y) &&
//             (a.br.y >= b.ul.y));

    bool dInGtX = (d.ul.x >= gt.ul.x) && (d.ul.x <= gt.br.x);
    bool GtInDX = (gt.ul.x >= d.ul.x) && (gt.ul.x <= d.br.x);
    bool dInGtY = (d.ul.y >= gt.ul.y) && (d.ul.y <= gt.br.y);
    bool GtInDY = (gt.ul.y >= d.ul.y) && (gt.ul.y <= d.br.y);
    
    return (dInGtX || GtInDX) && (dInGtY || GtInDY);
}


bool MotionEvaluation::intersectionOverUnionCriterion(const t_annotation& detection, const t_annotation& gt, const float& p)
{
    const float & intersectionArea = overlappingArea(detection, gt);
    const float unionArea = area(detection) + area(gt) - intersectionArea;
    
    const float intersectionOverUnion = intersectionArea / unionArea;

//     cout << "intersectionArea = " << intersectionArea << ", unionArea = " << unionArea << ", intersectionOverUnion = " << intersectionOverUnion << ", ";
    
    return intersectionOverUnion > p;
}

void MotionEvaluation::countErrors(const float& detectionThresh, const vector< t_annotation >& gt, const vector< t_annotation >& detections,
                                   uint32_t& tp, uint32_t& fn)
{
    const float p = 0.5f;
    
    vector< t_annotation > tmpGT(gt.size()), tmpDetections(detections.size());
    copy(gt.begin(), gt.end(), tmpGT.begin());
    copy(detections.begin(), detections.end(), tmpDetections.begin());
    
    typedef vector< t_annotation >::iterator annIt;
    for (annIt itGT = tmpGT.begin(); itGT != tmpGT.end(); itGT++) {
        const t_annotation & gtAnnotation = *itGT;

        bool found = false;
        for (annIt itDet = tmpDetections.begin(); itDet != tmpDetections.end(); itDet++) {
            const t_annotation & detectionAnnotation = *itDet;
            
            if (doOverlap(detectionAnnotation, gtAnnotation) &&
                intersectionOverUnionCriterion(detectionAnnotation, gtAnnotation, p)) {
                    
                    itGT = tmpGT.erase(itGT);
                    itDet = tmpDetections.erase(itDet);
                    
                    itGT--;
                    itDet--;
                    
                    break;
            }
        }
    }
    
    tp = detections.size() - tmpDetections.size();
    fn = tmpGT.size();
}

void MotionEvaluation::getAnnotationsFromTracks(const StixelsTracker::t_historic & historic, const uint32_t& idx, const uint32_t & currentFrame,
                                                vector< t_annotation >& detections)
{
    vector< t_annotation > gt = m_annotations[currentFrame - idx];
    detections.reserve(gt.size());
    
    vector <Stixel3d> evolution;
    evolution.reserve(historic[0].size());
    for (uint32_t i = 0; i < historic[idx].size(); i++) {
        Stixel3d stixel = historic[idx][i];
        for (int32_t j = idx - 1; j >= 0; j--) {
            if (stixel.valid_forward_delta_x)
                stixel = historic[j][stixel.forward_delta_x];
            else {
                stixel.x = -1;
                break;
            }
        }
        evolution.push_back(stixel);
    }
    
    BOOST_FOREACH(const t_annotation &gtAnnotation, gt) {
        t_annotation currDetection;
        currDetection.ul = cv::Point2i(INT_MAX, INT_MAX);
        currDetection.br = cv::Point2i(INT_MIN, INT_MIN);
        
        for (uint32_t x = gtAnnotation.ul.x; x <= gtAnnotation.br.x; x++) {
            const Stixel3d & currStixel = evolution[x];
            
            if (currStixel.x != -1) {
                
                if (currStixel.x < currDetection.ul.x) currDetection.ul.x = currStixel.x;
                if (currStixel.x > currDetection.br.x) currDetection.br.x = currStixel.x;
            
                if (currStixel.top_y < currDetection.ul.y) currDetection.ul.y = currStixel.top_y;
                if (currStixel.bottom_y > currDetection.br.y) currDetection.br.y = currStixel.bottom_y;
            }
        }
        
//         uint32_t middleX = (gtAnnotation.ul.x + gtAnnotation.br.x) / 2.0;
//         currDetection.ul.x = middleX;
//         currDetection.br.x = middleX + 1;
//         currDetection.ul.y = evolution[middleX].top_y;
//         currDetection.br.y = evolution[middleX].bottom_y;
        
        detections.push_back(currDetection);
    }
}

}