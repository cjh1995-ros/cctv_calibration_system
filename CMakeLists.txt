cmake_minimum_required(VERSION 2.8.3)
project(diverse_camera_model_calibration)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
find_package(OpenCV REQUIRED)
find_package( Ceres REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRECTORIES}
  ${CERES_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
)
