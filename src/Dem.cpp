#include "Dem.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

/* ---------------- 对外入口 ---------------- */
Dem::Dem(const std::string& tiff_path, const std::string& root_out)
    : root_out_(root_out)
{
    namespace fs = std::filesystem;
    const std::string root = root_out + "/DEM";
    fs::create_directories(root);
    out_img_dir_ = root + "/out_image_file";
    out_txt_dir_ = root + "/out_txt_file";
    fs::create_directories(out_img_dir_);
    fs::create_directories(out_txt_dir_);

    readTiff_(tiff_path);      // 读
    decodeToMeters_();         // 解码
    exportResults_();          // 导出
}

/* ---------------- 读 TIFF ---------------- */
void Dem::readTiff_(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) throw std::runtime_error("Cannot read TIFF: " + path);
    if (img.channels() != 1) throw std::runtime_error("TIFF must be single-channel");
    const int d = img.depth();
    if (d != CV_32F && d != CV_32S)
        throw std::runtime_error("TIFF depth must be CV_32F or CV_32S");
    raw_ = img.clone();
    std::cout << "Raw DEM loaded: " << width() << " x " << height() << std::endl;
    std::cout << "OpenCV raw type: " << d << "\n";
}

/* ---------------- 解码到米 ---------------- */
void Dem::decodeToMeters_()
{
    const int d = raw_.depth();
    dem_m_.create(raw_.size(), CV_64FC1);
    if (d == CV_32S) {
        double min_elev = 0, max_elev = 0;
        std::cout << "Raw is CV_32S (encoded). Enter elevation range:\n";
        std::cout << "MIN elevation (m): ";  std::cin >> min_elev;
        std::cout << "MAX elevation (m): ";  std::cin >> max_elev;
        if (max_elev <= min_elev)
            throw std::runtime_error("Max must be > min");
        min_elev_m_ = min_elev;
        delta_h_m_ = max_elev - min_elev;
        const double scale = delta_h_m_ / 4294967296.0;
        for (int y = 0; y < raw_.rows; ++y) {
            const int32_t* src = raw_.ptr<int32_t>(y);
            double* dst = dem_m_.ptr<double>(y);
            for (int x = 0; x < raw_.cols; ++x) {
                uint32_t u = static_cast<uint32_t>(src[x]);
                dst[x] = u * scale + min_elev_m_;
            }
        }
    }
    else if (d == CV_32F) {
        std::cout << "Raw is CV_32F (already meters). No manual range needed.\n";
        for (int y = 0; y < raw_.rows; ++y) {
            const float* src = raw_.ptr<float>(y);
            double* dst = dem_m_.ptr<double>(y);
            for (int x = 0; x < raw_.cols; ++x) {
                dst[x] = static_cast<double>(src[x]);
            }
        }
    }
    else {
        throw std::runtime_error("Unexpected depth in decodeToMeters_");
    }
}

/* ---------------- 导出结果 ---------------- */
void Dem::exportResults_()
{
    /* ---- 文本 ---- */
    const std::string txt_path = out_txt_dir_ + "/dem.txt";
    std::ofstream fs(txt_path);
    if (!fs) throw std::runtime_error("Cannot write: " + txt_path);
    fs << std::fixed << std::setprecision(3);
    for (int r = 0; r < dem_m_.rows; ++r) {
        const double* row = dem_m_.ptr<double>(r);
        for (int c = 0; c < dem_m_.cols; ++c) {
            fs << row[c] << (c + 1 < dem_m_.cols ? ' ' : '\n');
        }
    }
    /* ---- 图 ---- */
    savePng_(out_img_dir_ + "/raw_8u.png", to8U_(raw_));
    savePng_(out_img_dir_ + "/raw_16u.png", to16U_(raw_));

    std::cout << "\nOutputs:\n";
    std::cout << "  " << txt_path << "\n";
    std::cout << "  " << out_img_dir_ << "/raw_8u.png\n";
    std::cout << "  " << out_img_dir_ << "/raw_16u.png\n";
}

/* ---------- 工具函数 ---------- */
cv::Mat Dem::normalize01_(const cv::Mat& src)
{
    cv::Mat f; src.convertTo(f, CV_64F);
    double mn, mx; cv::minMaxLoc(f, &mn, &mx);
    double denom = (mx > mn) ? (mx - mn) : 1.0;
    return (f - mn) * (1.0 / denom);
}
cv::Mat Dem::to8U_(const cv::Mat& src)
{
    cv::Mat n = normalize01_(src);
    cv::Mat out; n.convertTo(out, CV_8U, 255.0); return out;
}
cv::Mat Dem::to16U_(const cv::Mat& src)
{
    cv::Mat n = normalize01_(src);
    cv::Mat out; n.convertTo(out, CV_16U, 65535.0); return out;
}
void Dem::savePng_(const std::string& path, const cv::Mat& img)
{
    if (!cv::imwrite(path, img)) throw std::runtime_error("imwrite failed: " + path);
}