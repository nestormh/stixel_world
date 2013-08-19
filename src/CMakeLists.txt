add_executable(testStixelWorld 
    stixelsapplication.cpp 
    groundestimator.cpp 
    rectification.cpp
#     doppia/GroundPlane.cpp
#     doppia/IrlsLinesDetector.cpp
    main.cpp
)
  
include_directories(testStixelWorld
#     GPULibSVM
    ${EIGEN3_INCLUDE_DIR}
#     ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
#     emon.a
#     ${CUDA_INCLUDE_DIRS}
    /home/nestor/Dropbox/KULeuven/projects/doppia/src/applications/stixel_world_lib/
    /home/nestor/Dropbox/KULeuven/projects/doppia/src/
    /home/nestor/Dropbox/KULeuven/projects/doppia/libs/
)

target_link_libraries(testStixelWorld
  ${EIGEN3_LIBRARIES}
#   ${PCL_LIBRARIES}
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
