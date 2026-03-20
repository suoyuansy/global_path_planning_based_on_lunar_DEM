#pragma once
#include <opencv2/core.hpp>
#include <string>

class Dem {
public:
    Dem(const std::string& tiff_path,
        const std::string& root_out,
        bool export_file_flag = true,
        bool simple_output_mode = false);//simple_output_mode简化输出模式，用作外部调用全局路径规划

    const cv::Mat& raw() const { return raw_; }
    const cv::Mat& demMeters() const { return dem_m_; }
    int width() const { return raw_.cols; }
    int height() const { return raw_.rows; }

private:
    cv::Mat raw_;
    cv::Mat dem_m_;
    double min_elev_m_ = 0;
    double delta_h_m_ = 0;

    std::string root_out_;
    std::string out_img_dir_;
    std::string out_txt_dir_;

    bool export_file_flag_ = true;
    bool simple_output_mode_ = false;

    void readTiff_(const std::string& path);
    void decodeToMeters_();
    void exportResults_();

    static cv::Mat normalize01_(const cv::Mat& src);
    static cv::Mat to8U_(const cv::Mat& src);
    static cv::Mat to16U_(const cv::Mat& src);
    static cv::Mat colorElevation_(const cv::Mat& dem_m);
    static void savePng_(const std::string& path, const cv::Mat& img);
};