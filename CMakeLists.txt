cmake_minimum_required(VERSION 3.27)
project(diverse_camera_model_calibration)

add_compile_options(-std=c++14)

# list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt" "/opt/homebrew/lib" "/usr/local/include")
set(CMAKE_PREFIX_PATH "/usr/local/include" "/opt/homebrew/opt" "/opt/homebrew/lib")
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

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
