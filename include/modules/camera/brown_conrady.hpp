#pragma once

#include "modules/camera/camera.h"
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <limits>
#include <cmath>



namespace DCMC
{


class BrownConrady : public CameraModel
{
public:
    explicit BrownConrady(const std::vector<double>& params) :
    {
        static_assert(params.size() == 6, "BrownConrady should have 6 parameters, fx, fy, cx, cy, k1, k2");
        params_ = params;
    }

    cv::Point2d const noexcept project(const cv::Point3d point) override
    {
        const double fx = params_[0];
        const double fy = params_[1];
        const double cx = params_[2];
        const double cy = params_[3];
        const double k1 = params_[4];
        const double k2 = params_[5];

        const double x = point.x / point.z;
        const double y = point.y / point.z;

        const double r2 = x * x + y * y;
        const double r4 = r2 * r2;

        const double radial = 1.0 + k1 * r2 + k2 * r4;

        const double u = fx * x * radial + cx;
        const double v = fy * y * radial + cy;

        return cv::Point2d(u, v);
    }

    cv::Point3d const noexcept unproject(const cv::Point2d pix) override
    {
        const double fx = params_[0];
        const double fy = params_[1];
        const double cx = params_[2];
        const double cy = params_[3];
        const double k1 = params_[4];
        const double k2 = params_[5];

        const double mx = (pix.x - cx) / fx;
        const double my = (pix.y - cy) / fy;

        const double rd = std::sqrt(mx * mx + my * my);
        const double ru = gauss_newton(rd);

        const double x = mx * ru / rd;
        const double y = my * ru / rd;

        return cv::Point3d(x, y, 1.0);
    }


    double gauss_newton(double rd) noexcept
    {
        double ru = rd;
        double init_diff = std::numeric_limits<double>::max();
        double step = 0.1;
        size_t max_i = 100;

        for (size_t i=0; i<max_i; i++)
        {
            diff = rd - ru * (1 + k1 * ru * ru + k2 * ru * ru * ru * ru);

            if ((i == max_i - 1) && (std::abs(diff) > 1e-4)) 
                std::cout << "Gauss-Newton iteration did not converge" << std::endl;

            if (init_diff > std::abs(diff)) step *= 1.2;
            else step *= 0.5;

            if (std::abs(diff) < 1e-4) break;

            init_diff = std::abs(diff);
            ru -= step * ru * (1 + k1 * ru * ru + k2 * ru * ru * ru * ru) / (1 + 3 * k1 * ru * ru + 5 * k2 * ru * ru * ru * ru);
        }

        return ru;
    }

    struct BrownConradyCeresReprojectionError : public BasicReprojectionError
    {
        BrownConradyCeresReprojectionError(cv::Point2d observed, cv::Point3d point):
            BasicReprojectionError(observed, point) {}

        template <typename T>
        bool operator() (const T* const intrinsic_, const T* const transform, T* residuals) override
        {
            const double fx = intrinsic_[0];
            const double fy = intrinsic_[1];
            const double cx = intrinsic_[2];
            const double cy = intrinsic_[3];
            const double k1 = intrinsic_[4];
            const double k2 = intrinsic_[5];

            T[3] P;
            T[3] point = {T(point_.x), T(point_.y), T(point_.z)};

            ceres::AngleAxisRotatePoint(transform, point, P);

            P[0] += transform[3];
            P[1] += transform[4];
            P[2] += transform[5];

            T x = P[0] / P[2];
            T y = P[1] / P[2];
            T one = T(1);

            T r2 = x * x + y * y;
            T r4 = r2 * r2;

            T radial = one + k1 * r2 + k2 * r4;

            T u = fx * x * radial + cx;
            T v = fy * y * radial + cy;

            residuals[0] = u - T(observed_.x);
            residuals[1] = v - T(observed_.y);

            return true;
        }

        static ceres::CostFunction* create(const cv::Point2d& obs, const cv::Point3d& pt)
        {
            return (new ceres::AutoDiffCostFunction<BrownConradyCeresReprojectionError, 2, 6, 6>(new BrownConradyCeresReprojectionError(obs, pt)));
        }

    };

}




} // namespace