#pragma once
#include <opencv2/core.hpp>

enum class PlanningStrategy {
    DistanceShortest = 0,   // g_p(θ)=0  距离最短
    SlopeCostMin = 1    // g_p(θ)=θ/20  坡度代价最小
};

struct CostResult {
    cv::Mat cost;       // CV_64FC1
    cv::Mat obstacle;   // CV_8UC1  (1=障碍, 0=可通行)
};

class TerrainCost {
public:
    // slope_deg: CV_64FC1 坡度(°)
    // theta_max_deg: 最大可通行坡度，20
    // inf_cost: 障碍代价
    static CostResult BuildCostFromSlope(const cv::Mat& slope_deg,PlanningStrategy strategy,double theta_max_deg = 20.0,double inf_cost = 1e10);
};
