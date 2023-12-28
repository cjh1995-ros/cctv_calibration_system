#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>




namespace DCMC
{


struct Frame
{
    explicit Frame(size_t id_, const std::string image_path) 
        :  id(id_)
        {
            image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            width = image.cols;
            height = image.rows;
        }

    std::vector<cv::Point2d> get_corners() const noexcept { return corners; }
    cv::Point2d get_corner(const int& id) const noexcept { return corners[id]; }
    int get_corner_num() const noexcept { return corners.size(); }
    cv::Vec6d get_transform() const noexcept { return transform; }
    void set_corners(const std::vector<cv::Point2d>& corners_) noexcept { corners = corners_; }
    void set_transform(const cv::Vec6d& transform_) noexcept { transform = transform_; }

    size_t                      id;
    int                         width;
    int                         height;
    cv::Mat                     image;
    std::vector<cv::Point2d>    corners;
    cv::Vec6d                   transform;
};



/// @brief Define is this good frame or not. It will be decided by below criteria
/// 1. Number of corners should be more than 0
/// 2. The ratio of convex per image area should be in [0.02, 0.25].
/// @param frame 
/// @return 
bool is_good_frame(const Frame& frame) noexcept
{
    if(frame.corners.size() == 0) return false;
    return true;
}

} // namespace