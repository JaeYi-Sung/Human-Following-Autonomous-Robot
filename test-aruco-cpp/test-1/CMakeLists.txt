# cmake_minimum_required(VERSION 3.0)

# set(CMAKE_BUILD_TYPE Debug)

# get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# string(REPLACE " " "_" ProjectId ${ProjectId})
# project(${ProjectId})

# set (CMAKE_CXX_STANDARD 11)

# file(GLOB SOURCES  *.cpp)

# add_executable(${PROJECT_NAME} ${SOURCES} )

# # 추가

# get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME) 

# 변경

get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)  

string(REPLACE " " "_" ProjectId ${ProjectId})

project(${ProjectId} C CXX)

set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
 
file(GLOB SOURCES  *.cpp)
 
add_executable(${PROJECT_NAME} ${SOURCES}  )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

