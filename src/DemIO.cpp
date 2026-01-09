#include "DemIO.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <iomanip>

static cv::Mat Normalize01(const cv::Mat& src) {
    if (src.empty() || src.channels() != 1) {
        throw std::runtime_error("Normalize01: src must be non-empty single-channel.");
    }
    cv::Mat f;
    src.convertTo(f, CV_64F);

    double mn, mx;
    cv::minMaxLoc(f, &mn, &mx);
    double denom = mx - mn;
    if (denom <= 0.0) denom = 1.0;

    return (f - mn) * (1.0 / denom);
}

cv::Mat DemIO::To8U_Linear(const cv::Mat& src) {
    cv::Mat n = Normalize01(src);
    cv::Mat out;
    n.convertTo(out, CV_8U, 255.0);
    return out;
}

cv::Mat DemIO::To16U_Linear(const cv::Mat& src) {
    cv::Mat n = Normalize01(src);
    cv::Mat out;
    n.convertTo(out, CV_16U, 65535.0);
    return out;
}

void DemIO::SaveImage(const std::string& path, const cv::Mat& img) {
    if (!cv::imwrite(path, img)) {
        throw std::runtime_error("SaveImage failed: " + path);
    }
}

void DemIO::ShowImage(const std::string& win, const cv::Mat& img, int wait_ms) {
    cv::namedWindow(win, cv::WINDOW_NORMAL);
    cv::imshow(win, img);
    cv::waitKey(wait_ms);
}


void DemIO::ExportDemToText(const Dem& dem, const std::string& out_path) {
    namespace fs = std::filesystem;

    // 自动创建父目录
    fs::path p(out_path);
    fs::create_directories(p.parent_path());

    const cv::Mat& m = dem.demMeters();   // 物理高程（单位：米）

    if (m.empty() || m.channels() != 1) {
        throw std::runtime_error("ExportDemToText: invalid DEM matrix.");
    }

    std::ofstream ofs(out_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + out_path);
    }

    ofs << std::fixed << std::setprecision(3);  // 保留 3 位小数

    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            ofs << m.at<double>(r, c);
            if (c < m.cols - 1) ofs << " ";
        }
        ofs << "\n";
    }

    ofs.close();
}


void DemIO::ExportAndPreview(const Dem& dem, const std::string& out_dir) {
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    // raw 预览（编码值）
    cv::Mat raw8 = To8U_Linear(dem.raw());
    cv::Mat raw16 = To16U_Linear(dem.raw());

    SaveImage(out_dir + "/raw_8u.png", raw8);
    SaveImage(out_dir + "/raw_16u.png", raw16);

    //ShowImage("RAW 8U", raw8, 1);
    //ShowImage("RAW 16U", raw16, 0);
}

