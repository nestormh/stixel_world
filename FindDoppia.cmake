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
  ${DOPPIA_BASE_PATH}/src/stereo_matching
  ${DOPPIA_BASE_PATH}/src/stereo_matching/ground_plane
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
  ${DOPPIA_BASE_PATH}/src/stereo_matching/stixels/ground_top_and_bottom.pb.cc

)

# set (DOPPIA_LIB ${DOPPIA_BASE_PATH}/src/applications/stixel_world_lib/build/libstixel_world.so)

# # From Doppia
set (doppia_root ${DOPPIA_BASE_PATH})
include(${doppia_root}/common_settings.cmake)
set(USE_GPU OFF CACHE BOOL "Should the GPU be used ?" FORCE)
# 
# # ----------------------------------------------------------------------
# # Setup required libraries
# 
pkg_check_modules(libpng REQUIRED libpng)
# pkg_check_modules(libjpeg REQUIRED libjpeg)
# pkg_check_modules(opencv REQUIRED opencv>=2.3)
# 
# set(opencv_LIBRARIES
#     opencv_core opencv_imgproc opencv_highgui opencv_ml
#     opencv_video opencv_features2d
#     opencv_calib3d
#     #opencv_objdetect opencv_contrib
#     opencv_legacy opencv_flann
#     opencv_gpu
#    ) # quick hack for opencv2.4 support
# 
find_package(Boost REQUIRED  
   COMPONENTS program_options filesystem system thread
)
# 
# # ----------------------------------------------------------------------
# # Setup link and include directories
# 
set(local_LIBRARY_DIRS
  "/usr/local/lib"
  "/users/visics/rbenenso/no_backup/usr/local/lib"
  "/usr/lib64"
  "/usr/lib64/atlas"
  "/usr/lib/sse2/atlas"
)
set(local_INCLUDE_DIRS
  "/users/visics/rbenenso/no_backup/usr/local/include"
  "/usr/include/eigen2/"
   "/usr/local/include/eigen2"
  "${doppia_root}/libs/cudatemplates/include"
)

link_directories(
  ${libpng_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  ${local_LIBRARY_DIRS}
)

include_directories(
  ${doppia_root}/libs
  ${doppia_root}/src
  ${libpng_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${local_INCLUDE_DIRS}
)
# 
# # ----------------------------------------------------------------------
# # Collect source files
# 
set(doppia_src "${doppia_root}/src")
set(doppia_stereo "${doppia_root}/src/stereo_matching")
# 
include_directories(${doppia_stereo}/ground_plane)  # for protoc plane3d.pb.h
# 
file(GLOB SrcCpp
  "./*.cpp"
  "${doppia_src}/*.cpp"
  "${doppia_src}/applications/*.cpp"
  
  "${doppia_src}/applications/stixel_world_lib/stixel_world_lib.cpp"

#   "${doppia_stereo}/*.cpp"
  "${doppia_stereo}/cost_volume/*CostVolume.cpp"
  "${doppia_stereo}/cost_volume/*CostVolumeEstimator*.cpp"
  "${doppia_stereo}/cost_volume/DisparityCostVolumeFromDepthMap.cpp"
  "${doppia_stereo}/cost_functions.cpp"
  "${doppia_stereo}/CensusCostFunction.cpp"
  "${doppia_stereo}/CensusTransform.cpp"
  "${doppia_stereo}/GradientTransform.cpp"
  "${doppia_stereo}/AbstractStereoMatcher.cpp"
  "${doppia_stereo}/AbstractStereoBlockMatcher.cpp"
  "${doppia_stereo}/SimpleBlockMatcher.cpp"
  "${doppia_stereo}/MutualInformationCostFunction.cpp"
  "${doppia_stereo}/ConstantSpaceBeliefPropagation.cpp"
  "${doppia_stereo}/qingxiong_yang/*.cpp"
  "${doppia_stereo}/SimpleTreesOptimizationStereo.cpp"
  "${doppia_stereo}/OpenCvStereo.cpp"

  "${doppia_stereo}/ground_plane/*.cpp"
  "${doppia_stereo}/ground_plane/*.cc"
  "${doppia_stereo}/stixels/*.cpp"
  "${doppia_stereo}/stixels/*.cc"
  "${doppia_stereo}/stixels/motion/*.cpp"

  "${doppia_src}/objects_detection/Detection2d.cpp"
  "${doppia_src}/objects_tracking/tracked_detections/TrackedDetection2d.cpp"

  "${doppia_src}/video_input/*.cpp"
  "${doppia_src}/video_input/AbstractVideoInput.cpp"
  "${doppia_src}/video_input/ImagesFromDirectory.cpp"
  "${doppia_src}/video_input/MetricCamera.cpp"
  "${doppia_src}/video_input/MetricStereoCamera.cpp"
  "${doppia_src}/video_input/VideoFromFiles.cpp"
  "${doppia_src}/video_input/calibration/*.c*"
  "${doppia_src}/video_input/preprocessing/*.cpp"
  "${doppia_src}/features_tracking/*.cpp"
  "${doppia_src}/image_processing/*.cpp"
  "${doppia_src}/drawing/gil/*.cpp"
)

file(GLOB HelpersCpp
  #"${doppia_src}/helpers/*.cpp"
  "${doppia_src}/helpers/data/*.c*"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/get_section_options.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/AlignedImage.cpp"
  "${doppia_src}/helpers/replace_environment_variables.cpp"
)

set (DOPPIA_CPP_FILES
  ${DOPPIA_CPP_FILES}
  ${SrcCpp}
  ${HelpersCpp}
)
# 
set (DOPPIA_LIB 
   ${DOPPIA_LIB}
   ${Boost_LIBRARIES} 
   protobuf pthread
   SDL X11 Xext #Xrandr
   gomp
   ${libpng_LIBRARIES} jpeg
   ${opencv_LIBRARIES}

   #csparse sparse spblas mv
   #lapack blas atlas

   #${google_perftools_LIBS}
   # faster malloc and non intrusive profiler
   # via http://google-perftools.googlecode.com
)