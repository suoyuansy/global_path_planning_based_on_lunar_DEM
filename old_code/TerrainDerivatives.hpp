#pragma once
#include <opencv2/core.hpp>

struct SlopeAspectResult {
    cv::Mat slope_deg;   // CV_64FC1, 坡度(°)
    cv::Mat aspect_deg;  // CV_64FC1, 坡向(°), [0,360)
};

class TerrainDerivatives {
public:
    // g: 栅格尺寸(米)，例如 1000m（CE7DEM_1km）或 1m
    static SlopeAspectResult ComputeSlopeAspect_3rdOrder(const cv::Mat& dem_m, double g);
};
