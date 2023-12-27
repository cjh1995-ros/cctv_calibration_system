#pragma once

#include <Eigen/Core>
#include <opencv2/core/core.hpp>


namespace DCMC
{

template <typename T>
class Camera
{
public:
    void project(T* point, T* result) const;
    void unproject(T* pixel, T* result) const;


private:
    T* camera_;
};



}