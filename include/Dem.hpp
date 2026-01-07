#pragma once
#include <opencv2/core.hpp>
#include <string>

class Dem {
public:
    void setRaw(const cv::Mat& raw);
    const cv::Mat& raw() const;

    void setDemMeters(const cv::Mat& dem_m);
    const cv::Mat& demMeters() const;

    // 完整的高程编码参数
    void setElevationEncoding(double min_elev_m, double delta_h_m);
    bool hasElevationEncoding() const;

    double minElevationM() const;
    double maxElevationM() const;
    double deltaHeightM() const;

    int width() const;
    int height() const;
    int cvType() const;

    std::string summary() const;

private:
    cv::Mat raw_;
    cv::Mat dem_m_;   // CV_64FC1

    int cv_type_ = -1;

    bool has_encoding_ = false;
    double min_elev_m_ = 0.0;
    double delta_h_m_ = 0.0;
};
