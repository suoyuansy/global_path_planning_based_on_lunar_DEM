#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

class PathPlanning_Local_API {
public:

    enum class PlanStatus {
        OK = 0,
        START_IS_OBSTACLE,
        GOAL_IS_OBSTACLE,
        START_AND_GOAL_ARE_OBSTACLES,
        NO_PATH_FOUND
    };

    struct PlanResult {
        PlanStatus status = PlanStatus::NO_PATH_FOUND;
        std::vector<cv::Point> path;
        cv::Mat costmap;   // 保存代价地图
    };

    // 从 txt 文件读取 DEM，返回 CV_64FC1
    static cv::Mat loadDEMFromTxt(const std::string& txt_path);

    // 从 txt 文件读取 costmap，返回 CV_64FC1
    static cv::Mat loadCostmapFromTxt(const std::string& txt_path);

    // 从 DEM 矩阵直接规划，内部会先计算 costmap
    static PlanResult planFromDEM(const cv::Mat& dem,const cv::Point& start,const cv::Point& goal, const double grid_size);
    
    // 直接从已计算好的 costmap 规划
    static PlanResult planFromCostmap(const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal);
    
    // 统一保存结果到文件
    // output_path 作为主输出文件路径：
    // 1. 保存状态/路径到 output_path
    // 2. 保存 costmap 到 output_path 同目录下的 costmap.txt
    static void saveResultToFile(const PlanResult& result,const std::string& output_path);
private:
    static bool isObstacle_(const cv::Mat& costmap, int x, int y);
};