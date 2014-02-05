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


#ifndef EXTENDEDVIDEOINPUTFACTORY_H
#define EXTENDEDVIDEOINPUTFACTORY_H

#include <video_input/VideoInputFactory.hpp>

namespace doppia {
class ExtendedVideoInputFactory : public VideoInputFactory
{
public:
    static boost::program_options::options_description get_args_options();
    static AbstractVideoInput* new_instance(const boost::program_options::variables_map &options);
};
}

#endif // EXTENDEDVIDEOINPUTFACTORY_H
