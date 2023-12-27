cmake_minimum_required(VERSION 3.27)
project(diverse_camera_model_calibration)

add_compile_options(-std=c++17)

set(CMAKE_PREFIX_PATH "/usr/local/include" "/opt/homebrew/opt" "/opt/homebrew/lib")
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

include(FetchContent)

FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(jsoncpp)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


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
  # jsoncpp_lib_static
  jsoncpp_lib
)
target_include_directories(${PROJECT_NAME} PRIVATE ${jsoncpp_SOURCE_DIR}/include)




enable_testing()

add_executable(
  test_brown_conrady
  src/test_brown_conrady.cpp
)
target_link_libraries(
  test_brown_conrady
  GTest::gtest_main
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
)


include(GoogleTest)
gtest_discover_tests(test_brown_conrady)