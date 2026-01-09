#pragma once
#include "Dem.hpp"
#include <string>

class DemIO {
public:
    static cv::Mat To8U_Linear(const cv::Mat& single_channel_any_depth);
    static cv::Mat To16U_Linear(const cv::Mat& single_channel_any_depth);

    static void SaveImage(const std::string& path, const cv::Mat& img);
    static void ShowImage(const std::string& win, const cv::Mat& img, int wait_ms);

    static void ExportDemToText(const Dem& dem, const std::string& out_path);
    static void ExportAndPreview(const Dem& dem, const std::string& out_dir);
};
