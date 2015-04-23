source_group("CUDA Source Files" FILES ${CUDALK_SRC_CU})
source_group("CUDA Header Files" FILES ${CUDALK_INCLUDE_DIRS})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)

set(CUDAGDB_OPTS -g -G)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

#################################################################
# stixels_world_node
#################################################################
message(CUDALK_SRC_CU ${CUDALK_SRC_CU})

add_executable(stixels_world_node 
#     ${CUDALK_SRC_CU}
    ${STIXEL_WORLD_SRC}
    ${LIBELAS_SRC_FILES}
    doppia/extendedvideoinputfactory.cpp 
    doppia/extendedvideofromfiles.cpp 
    kalmanfilter.cpp 
    oflowtracker.cpp
    stixelsapplicationros.cpp
    mainStixels.cpp
)

include_directories(stixels_world_node
#     ${CUDALK_INCLUDE_DIRS}
    ${LIBELAS_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}/pcl-1.7
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${STIXEL_WORLD_INCLUDE_DIRS}
    /opt/ros/indigo/include
)

target_link_libraries(stixels_world_node
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${STIXEL_WORLD_LIBRARIES}
  ${catkin_LIBRARIES}
)