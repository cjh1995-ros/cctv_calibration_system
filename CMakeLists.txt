cmake_minimum_required(VERSION 3.27)
project(diverse_camera_model_calibration)

add_compile_options(-std=c++17)

set(CMAKE_PREFIX_PATH "/usr/local/include" "/opt/homebrew/opt" "/opt/homebrew/lib")
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

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
