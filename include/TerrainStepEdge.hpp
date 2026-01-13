#pragma once
#include <opencv2/core.hpp>
#include <string>

class TerrainStepEdge {
public:
    TerrainStepEdge(const cv::Mat& dem_m,const std::string& root_out,double max_step_m = 0.4);   // 15 cm 默认台阶高

    const cv::Mat& step_obstacle() const { return step_obstacle_; }

private:
    cv::Mat dem_m_;
    double max_step_;
    std::string root_out_, out_txt_dir_, out_img_dir_;

    cv::Mat step_obstacle_;   // 0 可通行 / 1 阶梯障碍

    void computeRoberts_();
    void export_file_();

    static cv::Mat visualizeObs8U_(const cv::Mat& obs);
    static void    savePng_(const std::string& path, const cv::Mat& img);
    static void    exportText_(const cv::Mat& m, const std::string& path);
};