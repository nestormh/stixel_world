// Copyright (c) 2009-2012, KU Leuven All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// Neither the name of the KU Leuven nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// The source code and the binaries created using them can only be used for scientific research and education purposes. Any commercial use of this software is prohibited. For a different license, please contact the contributors directly.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL KU Leuven BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// The original version of this code can be found at https://bitbucket.org/rodrigob/doppia

#ifndef IRLSLINESDETECTOR_HPP
#define IRLSLINESDETECTOR_HPP

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <list>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace doppia {

/// Lines detection using Iteratively reweighted least squares
/// this class assume that the input image is binary, and that
/// there is one single dominant line
/// http://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
/// Tukey loss
/// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class IrlsLinesDetector
{
public:
    typedef Eigen::ParametrizedLine<float, 2> line_t;
    typedef std::vector<line_t> lines_t;
    
//     typedef  boost::gil::gray8c_view_t source_view_t;

    static boost::program_options::options_description get_args_options();

    IrlsLinesDetector(const boost::program_options::variables_map &options);

    IrlsLinesDetector(const int intensity_threshold,
                      const int num_iterations,
                      const float max_tukey_c,
                      const float min_tukey_c);
    ~IrlsLinesDetector();

    typedef cv::Point2f point_t;
    typedef std::vector< point_t > points_t;

    /// Iterative reweighted least squares estimation of the line
    /// given a set of points (x,y) and initial weights for each point
    /// @param lines contains the single estimated line
    void operator()(const points_t &points,
                    const Eigen::VectorXf &prior_points_weights,
                    lines_t &lines);

    /// Provide the best estimate available estimate for the line
    /// this method should be called right before each call to operator()
    void set_initial_estimate(const line_t &line_estimate);

    float compute_l1_residual();

protected:

    const int intensity_threshold, num_iterations;
    const float max_tukey_c, min_tukey_c;

    bool has_previous_line_estimate;
    line_t previous_line_estimate;

    Eigen::VectorXf b, x, w;
    Eigen::MatrixXf A;
};

} // namespace doppia

#endif // IRLSLINESDETECTOR_HPP
