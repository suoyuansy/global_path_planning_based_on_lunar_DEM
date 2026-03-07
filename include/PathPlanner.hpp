#pragma once
#include <opencv2/core.hpp>
#include <vector>

class PathPlanner {
public:
    enum class Method {
        AStar = 0
    };

    static std::vector<cv::Point> plan(Method method,const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal);
    static std::vector<cv::Point> planAStar(const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal);
    //可添加其它路径规划方法


private:
    static bool isObstacle_(const cv::Mat& costmap, int x, int y);
    static bool isSafe_(const cv::Mat& costmap, int x, int y, int safe_radius = 2);
};