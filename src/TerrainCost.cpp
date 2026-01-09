#include "TerrainCost.hpp"
#include <stdexcept>

CostResult TerrainCost::BuildCostFromSlope(const cv::Mat& slope_deg,PlanningStrategy strategy,double theta_max_deg,double inf_cost) 
{
    if (slope_deg.empty() || slope_deg.type() != CV_64FC1 || slope_deg.channels() != 1) {
        throw std::runtime_error("BuildCostFromSlope: slope_deg must be CV_64FC1 single-channel.");
    }
    if (theta_max_deg <= 0.0) {
        throw std::runtime_error("BuildCostFromSlope: theta_max_deg must be > 0.");
    }

    cv::Mat cost(slope_deg.rows, slope_deg.cols, CV_64FC1, cv::Scalar(0.0));
    cv::Mat obs(slope_deg.rows, slope_deg.cols, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < slope_deg.rows; ++y) {
        const double* s = slope_deg.ptr<double>(y);
        double* c = cost.ptr<double>(y);
        unsigned char* o = obs.ptr<unsigned char>(y);

        for (int x = 0; x < slope_deg.cols; ++x) {
            const double theta = s[x];

            if (theta >= theta_max_deg) {
                c[x] = inf_cost;
                o[x] = 1;
                continue;
            }

            // 可通行：根据策略计算 g_p(theta)
            if (strategy == PlanningStrategy::DistanceShortest) {
                c[x] = 0.0;
            }
            else { // SlopeCostMin
                c[x] = theta / theta_max_deg; // 0~1
            }
        }
    }

    return { cost, obs };
}
