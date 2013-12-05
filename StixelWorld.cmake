# Output: 
# STIXEL_WORLD_SRC -> Sources
# STIXEL_WORLD_INCLUDE_DIRS -> Includes
# STIXEL_WORLD_LIBRARIES -> Required libraries

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${STIXEL_WORLD_PATH})

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.49.0 COMPONENTS filesystem system program_options iostreams)
find_package(OpenCV  REQUIRED )
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)
find_package( Eigen3 REQUIRED )
find_package(PCL REQUIRED)
find_package(SDL REQUIRED)
find_package(PNG REQUIRED)
find_package(Doppia REQUIRED)

# TODO: Add as a configuration file
set(POLAR_CALIBRATION_INCLUDE_PATH ${POLAR_CALIBRATION_PATH})
set(POLAR_CALIBRATION_SRC 
    ${POLAR_CALIBRATION_PATH}/polarcalibration.cpp 
    ${POLAR_CALIBRATION_PATH}/calibration.pb.cc
    ${POLAR_CALIBRATION_PATH}/visualizePolarCalibration.cpp)

INCLUDE("${DENSE_TRACKER_PATH}/DenseTracker.cmake")

set(STIXEL_WORLD_SRC
    ${STIXEL_WORLD_PATH}/src/doppia/stixel3d.cpp
    ${STIXEL_WORLD_PATH}/src/stixelstracker.cpp 
    ${STIXEL_WORLD_PATH}/src/fundamentalmatrixestimator.cpp 
    ${STIXEL_WORLD_PATH}/src/utils.cpp
    ${STIXEL_WORLD_PATH}/src/stixelsapplication.cpp 
#     ${STIXEL_WORLD_PATH}/src/doppia/groundestimator.cpp 
    ${STIXEL_WORLD_PATH}/src/rectification.cpp
    ${STIXEL_WORLD_PATH}/src/doppia/extendedfaststixelworldestimator.cpp 
    ${STIXEL_WORLD_PATH}/src/doppia/extendedstixelworldestimatorfactory.cpp
    ${STIXEL_WORLD_PATH}/src/doppia/extendedfastgroundplaneestimator.cpp
    ${STIXEL_WORLD_PATH}/src/motionevaluation.cpp
#     ${STIXEL_WORLD_PATH}/src/main.cpp
    
    ${DOPPIA_CPP_FILES}
    ${POLAR_CALIBRATION_SRC}
    ${DENSETRACKER_CCFILES}
    ${DENSETRACKER_HFILES}
)

set(STIXEL_WORLD_INCLUDE_DIRS
    ${STIXEL_WORLD_PATH}/src
    ${STIXEL_WORLD_PATH}/src/doppia
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${POLAR_CALIBRATION_INCLUDE_PATH}
    emon.a
    /usr/include/pcl-1.7  # This line is just to help kdevelop to index PCL includes (remove)
    ${DOPPIA_INCLUDE_DIRS}
    ${DENSETRACKER_INCLUDE_DIRS}
)

set(STIXEL_WORLD_LIBRARIES
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${SDL_LIBRARY}
  ${PNG_LIBRARIES}
  ${DOPPIA_LIB}
  ${DENSETRACKER_LIBRARIES}
  emon
)