#pragma once
#include <opencv2/core.hpp>
#include <string>

class PathPlanning_Global_API {
public:
    struct BuildResult {
        cv::Mat dem;
        cv::Mat costmap;
    };

    static BuildResult buildFromTiff(const std::string& tiff_path,
        const std::string& output_dir,
        double grid_size);

    static void saveCostmapTxt(const cv::Mat& costmap, const std::string& txt_path);
    static void saveCostmapVis(const cv::Mat& costmap, const std::string& img_path);

private:
    static cv::Mat visualizeCostmap_(const cv::Mat& costmap);
};