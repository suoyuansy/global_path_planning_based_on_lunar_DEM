#pragma once
#include <opencv2/core.hpp>
#include <string>

class TerrainIO {
public:
    // 输出 CV_64FC1 到 txt（空格分隔）
    static void ExportMat64ToText(const cv::Mat& m64, const std::string& out_path, int precision = 3);

    // 坡度可视化：默认把 0~20° 线性映射到 0~255，>=20° 直接 255
    static cv::Mat VisualizeSlope8U(const cv::Mat& slope_deg, double theta_max_deg = 20.0);

    // 代价可视化：
    // - DistanceShortest: 可通行=0(黑)，障碍=255(白)
    // - SlopeCostMin: 可通行 cost(0~1)->0~255，障碍=255
    static cv::Mat VisualizeCost8U(const cv::Mat& cost64, const cv::Mat& obstacle8u, double inf_cost = 1e10);

    static void SaveGrayPng(const std::string& path, const cv::Mat& img8u);
};
