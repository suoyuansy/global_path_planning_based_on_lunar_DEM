#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

class PathPlanning {
public:
    explicit PathPlanning(const std::string& color_png_path);

private:
    //ÎÄ¼þÂ·¾¶
    std::string color_png_;
    std::string costmap_dir_ = "out_put/costmap";
    std::string costmap_file_;

    cv::Mat base_img_;
    cv::Mat display_;

    cv::Mat costmap_;       // CV_64FC1  1.0=ÕÏ°­

    cv::Point start_pt_;
    cv::Point goal_pt_;

    int  strategy_ = 0;     // 1 slope 2 rough 3 step 4 merge
    bool dist_first_ = false;
    bool expand_ = false;

    void consoleInput_();
    void loadColorImage_();
    void searchCostmapFile_();
    void loadCostmap_();
    void showAndInteract_();
    
    
    void redraw_();
    bool isObstacle_(int x, int y) const { return std::abs(costmap_.at<double>(y, x) - 1.0) < 1e-6; }
    std::vector<cv::Point> planAStar_() const;
    void savePath_(const std::vector<cv::Point>& path) const;
    static void mouseCallback_(int event, int x, int y, int flags, void* userdata);
    void onMouse_(int x, int y);

};