cmake_minimum_required(VERSION 3.24)
project(RedEyeCorrection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



add_executable(RedEyeCorrection main.cpp test_suite.h test_suite.h test_suite.cpp)

target_link_libraries(RedEyeCorrection ${OpenCV_LIBS})