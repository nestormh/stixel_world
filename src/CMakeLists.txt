add_executable(testStixelWorld 
    stixel3d.cpp
    stixelstracker.cpp 
    fundamentalmatrixestimator.cpp 
    utils.cpp
    stixelsapplication.cpp 
    groundestimator.cpp 
    rectification.cpp
    /home/nestor/Dropbox/KULeuven/projects/doppia/src/stereo_matching/stixels/motion/AbstractStixelMotionEstimator.cpp
    /home/nestor/Dropbox/KULeuven/projects/doppia/src/stereo_matching/stixels/motion/DummyStixelMotionEstimator.cpp
    /home/nestor/Dropbox/KULeuven/projects/doppia/src/drawing/gil/colors.cpp
#     doppia/GroundPlane.cpp
#     doppia/IrlsLinesDetector.cpp
    main.cpp
    ${POLAR_CALIBRATION_SRC}
)
  
include_directories(testStixelWorld
#     GPULibSVM
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${POLAR_CALIBRATION_INCLUDE_PATH}
#     emon.a
#     ${CUDA_INCLUDE_DIRS}
    /usr/include/pcl-1.7  # This line is just to help kdevelop to index PCL includes (remove)
    /home/nestor/Dropbox/KULeuven/projects/doppia/src/applications/stixel_world_lib/
    /home/nestor/Dropbox/KULeuven/projects/doppia/src/
    /home/nestor/Dropbox/KULeuven/projects/doppia/libs/
    /home/nestor/Dropbox/KULeuven/projects/doppia/src/stereo_matching/stixels/motion/
)

target_link_libraries(testStixelWorld
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
#   ${CUDA_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${SDL_LIBRARY}
#   /home/nestor/Dropbox/KULeuven/projects/doppia/src/applications/stixel_world_lib/build/libstixel_world_with_ui.so
  /home/nestor/Dropbox/KULeuven/projects/doppia/src/applications/stixel_world_lib/build/libstixel_world.so
  ${PNG_LIBRARIES}
#   ${Boost_
#   ${CGAL_CGAL_LIBRARY}
#   ${GMP_LIBRARIES}
#   ${CUDA_CURAND_LIBRARIES}
#   ${OPENGL_LIBRARIES}
#   ${GLUT_LIBRARIES}
#   emon
)

message(OPENGL_LIBRARIES ${OPENGL_LIBRARIES})
