# - Try to find Eigen3 lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Eigen3 3.1.2)
# to require version 3.1.2 or newer of Eigen3.
#
# Once done this will define
#
#  EIGEN3_FOUND - system has eigen lib with correct version
#  EIGEN3_INCLUDE_DIR - the eigen include directory
#  EIGEN3_VERSION - eigen version

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Copyright (c) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.

set (DOPPIA_BASE_PATH "/home/nestor/Dropbox/KULeuven/projects/doppia/")

set (DOPPIA_INCLUDE_DIRS 
  ${DOPPIA_BASE_PATH}/src
)

set (DOPPIA_CPP_FILES
  ${DOPPIA_BASE_PATH}/src/stereo_matching/stixels/motion/AbstractStixelMotionEstimator.cpp
  ${DOPPIA_BASE_PATH}/src/stereo_matching/stixels/motion/DummyStixelMotionEstimator.cpp
  ${DOPPIA_BASE_PATH}/src/drawing/gil/colors.cpp
  ${DOPPIA_BASE_PATH}/src/image_processing/IrlsLinesDetector.cpp
  ${DOPPIA_BASE_PATH}/src/stereo_matching/stixels/stixels.pb.cc
  ${DOPPIA_BASE_PATH}/src/objects_detection/detections.pb.cc
  ${DOPPIA_BASE_PATH}/src/objects_detection/detector_model.pb.cc
  ${DOPPIA_BASE_PATH}/src/helpers/data/DataSequenceHeader.pb.cc
  ${DOPPIA_BASE_PATH}/src/stereo_matching/ground_plane/plane3d.pb.cc
  ${DOPPIA_BASE_PATH}/src/stereo_matching/stixels/ground_top_and_bottom.pb.cc

)

set (DOPPIA_LIB ${DOPPIA_BASE_PATH}/src/applications/stixel_world_lib/build/libstixel_world.so)