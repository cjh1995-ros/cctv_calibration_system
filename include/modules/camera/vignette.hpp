#pragma once

#include <opencv2/core.hpp>
#include <cmath>
#include <vector>

namespace DCMC
{

using std::sqrt;

class VignetteModel
{
public:
    VignetteModel() = default;
    explicit VignetteModel(double v1, double v2, double v3, cv::Size img_size, cv::Point2d center):
        v1_(v1), v2_(v2), v3_(v3), img_size_(img_size), center_(center) 
        {
            // Who is biggest?
            //  1 --- 2
            //  |  c  |
            //  3 --- 4
            //
            const double x1 = center_.x;
            const double xx1 = x1 * x1;
            const double x2 = (img_size_.width - 1.0) - center_.x;
            const double xx2 = x2 * x2;
            const double y1 = center_.y;
            const double yy1 = y1 * y1;
            const double y2 = (img_size_.height - 1.0) - center_.y;
            const double yy2 = y2 * y2;

            max_radius_ = sqrt(std::max({xx1 + yy1, xx1 + yy2, xx2 + yy1, xx2 + yy2}));
        }

    /// @brief Get normalized radius at the location of pixel
    inline double GetNormedRadius(cv::Point2d& px) const noexcept
    {
        const double x = px.x - center_.x;
        const double y = px.y - center_.y;
        const double xx = x * x;
        const double yy = y * y;
        const double radius = sqrt(xx + yy);
        return radius / max_radius_;
    }

    /// @brief Get vignette factor at the normalized radius
    double GetVignetteFactor(double radius) const noexcept
    {
        const double r2 = radius * radius;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;

        return 1.0 + v1_ * r2 + v2_ * r4 + v3_ * r6;
    }

    /// @brief Get vignette factor with radius
    double GetVignetteFactor(cv::Point2d& px) const noexcept
    {
        const double radius = GetNormedRadius(px);
        return GetVignetteFactor(radius);
    }


    std::vector<double> GetVigneeteEstimate() const noexcept
    {
        std::vector<double> v;
        v.push_back(v1_);
        v.push_back(v2_);
        v.push_back(v3_);
        return v;
    }

    /// @brief Get vignette factor at the location of pixel
    void SetVignetteParameters(std::vector<double> vignette_model)
    {
        v1_ = vignette_model[0];
        v2_ = vignette_model[1];
        v3_ = vignette_model[2];
    }

    double v1() const noexcept { return v1_; }
    double v2() const noexcept { return v2_; }
    double v3() const noexcept { return v3_; }
    cv::Size cvsize() const noexcept { return img_size_; }
    cv::Point2d center() const noexcept { return center_; }

    double GetMaxRadius() const noexcept { return max_radius_; }

private:
    /// @brief Params for vignette model
    double v1_, v2_, v3_;

    /// @brief Image size
    cv::Size img_size_;

    /// @brief image center
    cv::Point2d center_;

    /// @brief max_radius
    double max_radius_;
};

} // namespace adso