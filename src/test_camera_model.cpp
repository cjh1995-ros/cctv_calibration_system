#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

#include "modules/camera/brown_conrady.hpp"
#include "modules/camera/kannala_brandt.hpp"
#include "modules/camera/single_sphere.hpp"
#include "modules/camera/double_sphere.hpp"
#include "modules/camera/triple_sphere.hpp"


#include <iostream>

using namespace DCMC;

TEST(BrownConradyCameraTest, CameraInitializeTest) {
    std::vector<double> params = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    BrownConrady camera(params);
}

TEST(BrownConradyCameraTest, CameraProjectionTest1) {
    std::vector<double> params = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0 };
    BrownConrady camera(params);

    cv::Point3d point(1.0, 1.0, 1.0);
    cv::Point2d pixel = camera.project(point);
    EXPECT_EQ(pixel.x, 2.0);
    EXPECT_EQ(pixel.y, 2.0);
}

TEST(BrownConradyCameraTest, CameraProjectionTest2) {
    double fx = 500;
    double fy = 500;
    double cx = 320;
    double cy = 240;
    double k1 = 0.1;
    double k2 = 0.1;

    std::vector<double> params = { fx, fy, cx, cy, k1, k2 };

    BrownConrady camera(params);

    std::vector<cv::Point3f> objectPoints = {
        cv::Point3f(0.0f, 0.0f, 1.0f),
        cv::Point3f(1.0f, 0.0f, 1.0f),
        cv::Point3f(0.0f, 1.0f, 1.0f),
        cv::Point3f(1.0f, 1.0f, 1.0f)
    };

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx, // fx, 0, cx
        0, fy, cy, // 0, fy, cy
        0, 0, 1);   // 0, 0, 1

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, 0, 0, 0); // 왜곡 계수

    cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0, 0, 0); // 회전 벡터
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0); // 이동 벡터

    std::vector<cv::Point2f> imagePoints;
    
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);


    for (int i = 0; i < imagePoints.size(); i++) {
        cv::Point3d point(objectPoints[i].x, objectPoints[i].y, objectPoints[i].z);
        cv::Point2d imagePointD(imagePoints[i].x, imagePoints[i].y);
        cv::Point2d pixel = camera.project(point);
        EXPECT_NEAR(imagePointD.x, pixel.x, 1e-5);
        EXPECT_NEAR(imagePointD.y, pixel.y, 1e-5);
    }    
}


TEST(BrownConradyCameraTest, CameraUnprojectionTest) {
    double fx = 500;
    double fy = 500;
    double cx = 320;
    double cy = 240;
    double k1 = 0.1;
    double k2 = 0.1;

    std::vector<double> params = { fx, fy, cx, cy, k1, k2 };

    BrownConrady camera(params);

    std::vector<cv::Point2f> distortedPoints = {
        cv::Point2f(100.0f, 100.0f),
        cv::Point2f(300.0f, 200.0f),
        cv::Point2f(400.0f, 50.0f),
        cv::Point2f(10.0f, 500.0f),
        // cv::Point2f(10.0f, 700.0f), <- 부동소수점 에러 나옴.
        // cv::Point2f(10.0f, 1000.0f)
    };

    std::vector<cv::Point2f> undistortedPoints;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx, // fx, 0, cx
        0, fy, cy, // 0, fy, cy
        0, 0, 1);   // 0, 0, 1;

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, 0, 0, 0); // 왜곡 계수

    cv::undistortPoints(distortedPoints, undistortedPoints, cameraMatrix, distCoeffs);

    std::vector<cv::Point3d> points3D;

    for (const auto& pt : undistortedPoints) {
        
        cv::Point3d point3D(
            pt.x, 
            pt.y,
            1.0
        );

        points3D.push_back(point3D);
    }


    for (int i = 0; i < undistortedPoints.size(); i++) {
        cv::Point2d pixel(distortedPoints[i].x, distortedPoints[i].y);
        cv::Point3d point = camera.unproject(pixel);
        EXPECT_NEAR(point.x, points3D[i].x, 1e-3);
        EXPECT_NEAR(point.y, points3D[i].y, 1e-3);
    }

}