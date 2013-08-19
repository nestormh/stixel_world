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

#include "IrlsLinesDetector.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>

#include <boost/foreach.hpp>

#include <utility>
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace doppia {

using namespace std;
using namespace boost;

typedef IrlsLinesDetector::points_t points_t;
typedef Eigen::ParametrizedLine<float, 2> line_t;
typedef vector<line_t> lines_t;

IrlsLinesDetector::IrlsLinesDetector(const int intensity_threshold_,
                                     const int num_iterations_,
                                     const float max_tukey_c_,
                                     const float min_tukey_c_)
    : intensity_threshold(intensity_threshold_),
      num_iterations(num_iterations_),
      max_tukey_c(max_tukey_c_), min_tukey_c(min_tukey_c_),
      has_previous_line_estimate(false)
{
    assert(max_tukey_c >= min_tukey_c);
    return;
}


IrlsLinesDetector::~IrlsLinesDetector()
{
    // nothing to do here
    return;
}

void set_A_and_b(const points_t &points, const int &num_points, Eigen::MatrixXf &A, Eigen::VectorXf &b)
{
    A.setOnes(num_points, 2);
    b.setZero(num_points);

    int i=0;
    BOOST_FOREACH(points_t::const_reference point, points)
    {
        A(i, 0) = point.x; // x coordinate value
        b(i) = point.y; // y coordinate value
        i+=1;
    } // end of "for each point"

    return;
}

/*
void compute_initial_estimate(const points_t &points, const int &num_points)
{

    return;
}
*/

void x_to_line(const Eigen::VectorXf &x, line_t &the_line)
{
    assert(x.size() == 2);
    the_line.direction()(0) = x(0);
    the_line.origin()(0) = x(1);
    return;
}


void line_to_x(const line_t &the_line, Eigen::VectorXf &x)
{
    assert(x.size() == 2);
    x(0) = the_line.direction()(0);
    x(1) = the_line.origin()(0);
    return;
}


/// as defined in
/// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
inline float tukey_weight(const float x, const float c)
{
    assert(c > 0);
    float weight = 0;

    if(std::abs(x) <= c)
    {
        const float x_div_c = x/c;
        const float delta = 1 - x_div_c*x_div_c;
        weight = delta*delta;
    }
    else
    {
        // weight = 0;
    }

    return weight;
}

// tukey_c is in [pixels]
void recompute_weights(const Eigen::MatrixXf &A,
                       const Eigen::VectorXf &b,
                       const Eigen::VectorXf &x,
                       const bool use_horizontal_distance,
                       const float tukey_c,
                       Eigen::VectorXf &w)
{
    // we assume that w is already initialized
    assert(b.rows() == w.rows());

    if(use_horizontal_distance == false)
    {
        std::runtime_error("recompute_weights with vertical distance it not yet implemented");
    }

    Eigen::VectorXf &horizontal_error = w;

    horizontal_error = (b - A*x);

    for(int i=0; i < b.rows(); i+=1)
    {
        w(i) = tukey_weight(horizontal_error(i), tukey_c);
    }

    return;
}

/// Provide the best estimate available estimate for the line
void IrlsLinesDetector::set_initial_estimate(const line_t &line_estimate)
{
    previous_line_estimate = line_estimate;
    has_previous_line_estimate = true;
    return;
}

/// implementation based on the following tutorial
/// http://graphics.stanford.edu/~jplewis/lscourse/SLIDES.pdf
void IrlsLinesDetector::operator()(const points_t &points,
                                   const Eigen::VectorXf &prior_points_weights,
                                   lines_t &lines)
{
    lines.clear();


    const int num_points = prior_points_weights.rows();
    assert(points.size() == static_cast<size_t>(num_points));

    if(false and points.size() != static_cast<size_t>(num_points))
    {
        printf("num_points count is wrong %zi != %i\n", points.size(), num_points);
        throw std::runtime_error("num_points count is wrong");
    }

    if(num_points < 2)
    {
        // not enough points to compute a line
        // returning zero lines indicates that something went wrong
        return;
    }

    // b=Ax

    set_A_and_b(points, num_points, A, b);
    x.setZero(2); // resize x
    w.setOnes(num_points); // resize w

    Eigen::MatrixXf new_A = A;
    Eigen::VectorXf new_b = b;

    // find initial line estimate --
    if(has_previous_line_estimate)
    {
        line_to_x(previous_line_estimate, x);
    }
    else
    {
        new_A = prior_points_weights.asDiagonal() * A;
        new_b = prior_points_weights.asDiagonal() * b;

        //const bool solved = new_A.svd().solve(new_b, &x); // Eigen2 form
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(new_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        x = svd.solve(new_b);
        const bool solved = true;
        if(solved == false)
        {
            throw std::runtime_error("IrlsLinesDetector::operator() Failed to initial line estimate");
        }
    }
    has_previous_line_estimate = false; // set false for next operator() call

    // iterative reweighted least squares --

    float tukey_c = max_tukey_c; // [pixels]
    const float tukey_c_step = (max_tukey_c - min_tukey_c) / std::max(1, num_iterations - 1); // [pixels]

    //cout << "Initial x" << endl << x << endl;
    for(int i=0; i < num_iterations; i+=1, tukey_c -= tukey_c_step)
    {
        // measure distance between points and line -
        // ( use horizontal or vertical distance )
        const bool use_horizontal_distance = true;
        recompute_weights(A, b, x, use_horizontal_distance, tukey_c, w);
        w = w.cwiseProduct(prior_points_weights);

        // setup the weighted least square estimate -
        new_A = w.asDiagonal() * A;
        new_b = w.asDiagonal() * b;

        // solve weighted least square estimate -        
        //const bool solved = new_A.svd().solve(new_b, &x); // Eigen2 form
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(new_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        x = svd.solve(new_b);
        const bool solved = true;
        if(solved == false)
        {
            printf("IrlsLinesDetector::operator() Failed to solve reweighted line estimate at iteration %i\n",
                   i);
            throw std::runtime_error("IrlsLinesDetector::operator() Failed to solve reweighted line estimate");
        }

        // cout << "x at iteration " << i << endl << x << endl;
        // cout << "tukey_c at iteration " << i << endl << tukey_c << endl;
    } // end of "for each iteration"


    // IrlsLinesDetector estimates only a single line --
    {
        line_t the_line;
        x_to_line(x, the_line);

        lines.push_back(the_line);
    }

    return;
}

float IrlsLinesDetector::compute_l1_residual()
{
    const float residual = ((A*x - b).array() * w.array()).abs().sum();
    return residual;
}

} // namespace doppia
