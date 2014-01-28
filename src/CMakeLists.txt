#################################################################
# stixels_world_node
#################################################################
add_executable(stixels_world_node
    ${STIXEL_WORLD_SRC}
    stixelsapplicationros.cpp
    mainStixels.cpp
)

include_directories(stixels_world_node
    ${PCL_INCLUDE_DIRS}/pcl-1.7
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${STIXEL_WORLD_INCLUDE_DIRS}
    /opt/ros/hydro/include
)

target_link_libraries(stixels_world_node
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${STIXEL_WORLD_LIBRARIES}
  ${catkin_LIBRARIES}
)