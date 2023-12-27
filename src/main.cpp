#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>

#include <ceres/ceres.h>
#include <ceres/rotation.h>



double estimate_focal(const std::vector<std::vector<cv::Point2d>>& pixels, const cv::Point2d opt_center, const cv::Size chessboard_num)
{
    double focal_ = 0;

    double cx_ = opt_center.x;
    double cy_ = opt_center.y;

    std::vector<std::vector<cv::Point2d>> pixels_center;
    for(int i = 0; i < pixels.size(); i++)
    {
        std::vector<cv::Point2d> pixels_tmp;
        for(int j = 0; j < pixels[i].size(); j++)
        {
            cv::Point2d p(pixels[i][j].x - cx_, pixels[i][j].y - cy_);
            pixels_tmp.push_back(p);
        }
        pixels_center.push_back(pixels_tmp);
    }
    int total_num = 0;
    for(int k = 0; k < pixels_center.size(); k++)
    {
        if(pixels_center[k].size() == 0) continue;
        for(int i = 0; i < chessboard_num.height; i++)
        {
            cv::Mat P(cv::Size(4, chessboard_num.width), CV_64F);
            for(int j = 0; j < chessboard_num.width; j++)
            {
                double x = pixels_center[k][i*chessboard_num.width+j].x;
                double y = pixels_center[k][i*chessboard_num.width+j].y;
                P.at<double>(j, 0) = x;
                P.at<double>(j, 1) = y;
                P.at<double>(j, 2) = 0.5;
                P.at<double>(j, 3) = -0.5*(x*x+y*y);
            }
            cv::Mat C;
            cv::SVD::solveZ(P, C);
            double c1 = C.at<double>(0);
            double c2 = C.at<double>(1);
            double c3 = C.at<double>(2);
            double c4 = C.at<double>(3);
            double t = c1*c1 + c2*c2 + c3*c4;
            if(t < 0) continue;
            double d = std::sqrt(1/t);
            double nx = c1 * d;
            double ny = c2 * d;
            if(nx*nx+ny*ny > 0.95) continue;
            double nz = std::sqrt(1-nx*nx-ny*ny);
            double gamma = fabs(c3*d/nz);
            focal_ += gamma;
            total_num++;
        }
    }

    focal_ /= total_num;

    return focal_;
}



// Test for Double sphere camera model calibration
struct ReprojectionError
{
    // 고정 변수
    cv::Point2d observed_;
    cv::Point3d point_;

    ReprojectionError(cv::Point2d observed, cv::Point3d point) : observed_(observed), point_(point) {}

    template <typename T>
    bool operator() (const T* const camera, const T* const transform, T* residuals) const
    {
        const T& fx = camera[0];
        const T& fy = camera[1];
        const T& cx = camera[2];
        const T& cy = camera[3];
        const T& alpha = camera[4];
        const T& xi = camera[5];

        T p[3]; // p is point in camera coordination
        T point[3] = {T(point_.x), T(point_.y), T(point_.z)};
        ceres::AngleAxisRotatePoint(transform, point, p);

        p[0] += transform[3];
        p[1] += transform[4];
        p[2] += transform[5];

        T x = p[0];
        T y = p[1];
        T z = p[2];

        T xx = x * x;
        T yy = y * y;
        T zz = z * z;

        T r2 = xx + yy;

        T d1_2 = r2 + zz;
        T d1 = ceres::sqrt(d1_2);

        T w1 = alpha > (T)(0.5) ? ((T)(1) - alpha) / alpha : alpha / ((T)(1) - alpha);
        T w2 = (w1 + xi) / ceres::sqrt( (T)(2) * w1 * xi + xi * xi + (T)(1) );

        bool is_valid = ( z > -w2 * d1);
        
        T k = xi * d1 + z;
        T k2 = k * k;

        T d2_2 = r2 + k2;
        T d2 = ceres::sqrt(d2_2);

        T norm = alpha * d2 + ((T)(1) - alpha) * k;

        T mx = x / norm;
        T my = y / norm;

        residuals[0] = fx * mx + cx - observed_.x;
        residuals[1] = fy * my + cy - observed_.y;

        return is_valid;
    }

    static ceres::CostFunction* create(const cv::Point2d& obs, const cv::Point3d& pt)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 6>(new ReprojectionError(obs, pt)));
    }
};


namespace fs = std::__fs::filesystem;

int main(int argc, char** argv) {
    int board_width = 9;
    int board_height = 6;
    double indent = 0.05;
    cv::Size board_size = cv::Size(board_width, board_height);

    // Initialize object points
    std::vector<cv::Point3d> object_points;
    object_points.reserve(board_width * board_height);

    for (int i = 0; i < board_height; ++i) {
        for (int j = 0; j < board_width; ++j) {
            object_points.emplace_back(j + indent, i + indent, 0.0);
        }
    }

    // Read images
    std::vector<cv::Mat> images;
    std::string data_path = "../data/left/";

    if (fs::exists(data_path) && fs::is_directory(data_path)) 
    {
        std::cout << "Data path exists." << std::endl;
        for (const auto& entry: fs::directory_iterator(data_path)) 
        {
            auto filename = entry.path().filename().string();
            if (entry.path().extension() == ".png")
            {
                images.emplace_back(cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE));
            }
        }
        std::cout << "Found " << images.size() << " images." << std::endl;
    } else {
        std::cout << "Data path does not exist." << std::endl;
    }

    if (images.empty()) {
        std::cout << "No images found." << std::endl;
        return -1;
    }

    // Find chessboard corners and Initialize camera poses
    double fx = images[0].cols / 2.0;
    double fy = images[0].rows / 2.0;
    double cx = (images[0].cols - 1.0) / 2.0;
    double cy = (images[0].rows - 1.0) / 2.0;

    double alpha = 0.01;
    double xi = 0.01;

    double camera[6] = {fx, fy, cx, cy, alpha, xi};

    std::vector<std::vector<cv::Point2d>> corners;
    std::vector<cv::Vec6d> camera_poses(images.size());

    for (size_t i = 0; i < images.size(); i++) {
        std::vector<cv::Point2f> corner;
        bool found = cv::findChessboardCorners(images[i], board_size, corner);

        if (found) {
            // Refine corner positions
            cv::cornerSubPix(images[i], corner, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1));
            
            // Insert corners
            std::vector<cv::Point2d> corner_d;
            corner_d.reserve(corner.size());
            for (const auto& pt: corner) {
                corner_d.emplace_back(pt.x, pt.y);
            }
            corners.emplace_back(corner_d);

            // Initialize camera poses
            cv::Mat rvec, tvec;
            cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            cv::solvePnP(object_points, corner, K, cv::Mat(), rvec, tvec);
            
            cv::Vec6d pose;
            pose[0] = rvec.at<double>(0, 0);
            pose[1] = rvec.at<double>(1, 0);
            pose[2] = rvec.at<double>(2, 0);
            pose[3] = tvec.at<double>(0, 0);
            pose[4] = tvec.at<double>(1, 0);
            pose[5] = tvec.at<double>(2, 0);

            camera_poses[i] = pose;
        } else {
            std::cout << "Could not find chessboard corners." << std::endl;
        }
    }

    // Estimate focal length by TSCM method
    cv::Point2d opt_center(cx, cy);
    double focal = estimate_focal(corners, opt_center, board_size);

    std::cout << "Focal by TSCM: " << focal << std::endl;

    // Optimize with ceres
    ceres::Problem problem;

    for (size_t j = 0; j < corners.size(); j++) {
        for ( size_t i = 0; i < corners[j].size(); i++) {
 
            ceres::CostFunction* cost_function = ReprojectionError::create(corners[j][i], object_points[i]);
            double* pose = (double*) (&camera_poses[j]);

            problem.AddResidualBlock(cost_function, nullptr, camera, pose);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);


    std::cout << "Initial camera parameters: " << std::endl;
    std::cout << "fx: " << fx << ", fy: " << fy << ", cx: " << cx << ", cy: " << cy << ", alpha: " << alpha << ", xi: " << xi << std::endl;
    std::cout << "Final camera parameters: " << std::endl;
    std::cout << "fx: " << camera[0] << ", fy: " << camera[1] << ", cx: " << camera[2] << ", cy: " << camera[3] << ", alpha: " << camera[4] << ", xi: " << camera[5] << std::endl;


    return 0;
}