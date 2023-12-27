#pragma once

#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <vector>



namespace DCMC
{




template <typename T>
void convertFromWorldToCamera(const T* rotation, const T* translation, const T* point, T* result)
{
    T p[3];
    ceres::AngleAxisRotatePoint(rotation, point, p);

    result[0] = p[0] + translation[0];
    result[1] = p[1] + translation[1];
    result[2] = p[2] + translation[2];
}

template <typename T>
void convertFrom3dToNormedPlane(const T* point, T* result)
{
    result[0] = point[0] / point[2];
    result[1] = point[1] / point[2];
}

template <typename T>
void convertFromCVToCeres(const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, T* result)
{
    result[0] = camera_matrix.at<double>(0, 0);
    result[1] = camera_matrix.at<double>(1, 1);
    result[2] = camera_matrix.at<double>(0, 2);
    result[3] = camera_matrix.at<double>(1, 2);
    result[4] = distortion_coefficients.at<double>(0, 0);
    result[5] = distortion_coefficients.at<double>(0, 1);
    result[6] = distortion_coefficients.at<double>(0, 2);
    result[7] = distortion_coefficients.at<double>(0, 3);
}

template <typename T>
void convertFromCeresToCV(const T* camera, cv::Mat& camera_matrix, cv::Mat& distortion_coefficients)
{
    camera_matrix.at<double>(0, 0) = camera[0];
    camera_matrix.at<double>(1, 1) = camera[1];
    camera_matrix.at<double>(0, 2) = camera[2];
    camera_matrix.at<double>(1, 2) = camera[3];
    distortion_coefficients.at<double>(0, 0) = camera[4];
    distortion_coefficients.at<double>(0, 1) = camera[5];
    distortion_coefficients.at<double>(0, 2) = camera[6];
    distortion_coefficients.at<double>(0, 3) = camera[7];
}

} // namespace