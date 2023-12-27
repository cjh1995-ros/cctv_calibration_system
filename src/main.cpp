#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct ReprojectionError
{
    // 고정 변수
    cv::Point2d observed_;
    cv::Point3d point_;

    ReprojectionError(double observed_x, double observed_y) : observed_(observed_x, observed_y) {}

    template <typename T>
    bool operator() (const T* const camera, const T* const transform, const T* const point, T* residuals) const
    {
        const T& fx = camera[0];
        const T& fy = camera[1];
        const T& cx = camera[2];
        const T& cy = camera[3];
        const T& k1 = camera[4];
        const T& k2 = camera[5];
        const T& p1 = camera[6];
        const T& p2 = camera[7];

        T p[3]; // p is point in camera coordination
        ceres::AngleAxisRotatePoint(transform, point, p);

        // To normalized plane
        const T& xp = p[0] / p[2];
        const T& yp = p[1] / p[2];

        // Transform 3d world point into camera coordination
        
    }
};




int main(int argc, char** argv) {
    int board_width = 9;
    int board_height = 6;
    cv::Size board_size = cv::Size(board_width, board_height);

    cv::Mat image = cv::imread("data/left/frame0000.png", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cout << "Could not read image file." << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> corners;

    bool found = cv::findChessboardCorners(image, board_size, corners);

    if (found) {
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1));
        
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

        cv::drawChessboardCorners(image, board_size, corners, found);
        
        cv::imshow("Image", image);
        cv::waitKey(0);
    } else {
        std::cout << "Could not find chessboard corners." << std::endl;}


    double camera_fx = 1.0, camera_fy = 1.0, camera_cx = 1.0, camera_cy = 1.0;
    double dist_coeffs[5] = {0.0, 0.0, 0.0, 0.0, 0.0};




    return 0;
}