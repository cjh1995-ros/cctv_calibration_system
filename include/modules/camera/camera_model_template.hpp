/**
 * File: camera_model_template.hpp
 * Author: Jun Hyeok Choi, dkwnsgur12@gmail.com
 * Date: 2023-12-27
 * Copyright (c) 2023 Jun Hyeok Choi. All rights reserved.
 * Description: This file is for camera model template. All camera model should be derived from this class.
 * 
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

namespace DCMC
{



/// @brief This class is for camera model template.
class CameraModel
{
public:
    CameraModel() = default;
    virtual ~CameraModel() {}

    virtual cv::Point2d const noexcept project(const cv::Point3d point) const = 0;
    virtual cv::Point3d const noexcept unproject(const cv::Point2d pixel) const = 0;
    
    std::vector<double> params() const noexcept { return params_; }
    void set_params(const std::vector<double> params) noexcept { params_ = params; }

    struct BasicReprojectionError 
    {
        cv::Point2d observed_;
        cv::Point3d point_;
        BasicReprojectionError(cv::Point2d observed, cv::Point3d point) : observed_(observed), point_(point) {}

        template <typename T>
        virtual bool operator() (const T* const intrinsic_, const T* const transform, T* residuals) const = 0;
    };

private:
    std::vector<double> params_;
};



}