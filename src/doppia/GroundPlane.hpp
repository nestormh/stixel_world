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


#ifndef GROUNDPLANE_HPP
#define GROUNDPLANE_HPP

#include <Eigen/Geometry>

#define IDX_X 0
#define IDX_Y 1
#define IDX_Z 2

namespace doppia {

/// Stores the ground plane in metric units
class GroundPlane : public Eigen::Hyperplane<float, 3>
{
public:
    /// Eigen::Hyperplane already defines (and documents) almost everything we need
    /// http://eigen.tuxfamily.org/dox-devel/classEigen_1_1Hyperplane.html

    /// Default constructor without initialization
    explicit GroundPlane();

    /// Constructs a plane from its normal \a n and distance to the origin \a d
    /// such that the algebraic equation of the plane is \f$ n \cdot x + d = 0 \f$.
    /// \warning the vector normal is assumed to be normalized.
    ////
    GroundPlane(const VectorType& n, Scalar d);

    /// picth and roll in [radians], height in [meters]
    void set_from_metric_units(const float pitch, const float roll, const float height);

    /// @returns the distance to the ground in [meters]
    const float &get_height() const;
    float &get_height();

    /// we assume no roll
    float get_pitch() const;

};

} // end of namespace doppia

#endif // GROUNDPLANE_HPP
