#pragma once
#include <opencv2/core.hpp>
#include <string>

class Dem {
public:
    Dem(const std::string& tiff_path, const std::string& root_out);
    /* 只读接口 */
    const cv::Mat& raw()       const { return raw_; }
    const cv::Mat& demMeters() const { return dem_m_; }
    int width()  const { return raw_.cols; }
    int height() const { return raw_.rows; }
private:
    cv::Mat raw_;          // 原始单通道 32S/32F
    cv::Mat dem_m_;        // 物理高程 CV_64FC1 单位米
    double min_elev_m_ = 0;
    double delta_h_m_ = 0;
    std::string root_out_; // 根目录
    std::string out_img_dir_;
    std::string out_txt_dir_;

    void readTiff_(const std::string& path);
    void decodeToMeters_();
    void exportResults_();

    static cv::Mat normalize01_(const cv::Mat& src);
    static cv::Mat to8U_(const cv::Mat& src);
    static cv::Mat to16U_(const cv::Mat& src);
    static void  savePng_(const std::string& path, const cv::Mat& img);
};