#include "TerrainIO.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

void TerrainIO::ExportMat64ToText(const cv::Mat& m64, const std::string& out_path, int precision) {
    if (m64.empty() || m64.channels() != 1) {
        throw std::runtime_error("ExportMat64ToText: input must be CV_64FC1 single-channel.");
    }

    namespace fs = std::filesystem;
    fs::path p(out_path);
    if (!p.parent_path().empty()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream ofs(out_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("ExportMat64ToText: failed to open file: " + out_path);
    }

    ofs << std::fixed << std::setprecision(precision);
    for (int r = 0; r < m64.rows; ++r) {
        const double* row = m64.ptr<double>(r);
        for (int c = 0; c < m64.cols; ++c) {
            ofs << row[c];
            if (c < m64.cols - 1) ofs << " ";
        }
        ofs << "\n";
    }
}

cv::Mat TerrainIO::VisualizeSlope8U(const cv::Mat& slope_deg, double theta_max_deg) {
    if (slope_deg.empty() || slope_deg.type() != CV_64FC1 || slope_deg.channels() != 1) {
        throw std::runtime_error("VisualizeSlope8U: slope_deg must be CV_64FC1 single-channel.");
    }
    if (theta_max_deg <= 0.0) theta_max_deg = 20.0;

    cv::Mat out(slope_deg.rows, slope_deg.cols, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < slope_deg.rows; ++y) {
        const double* s = slope_deg.ptr<double>(y);
        unsigned char* o = out.ptr<unsigned char>(y);
        for (int x = 0; x < slope_deg.cols; ++x) {
            double v = s[x];
            if (v < 0.0) v = 0.0;
            if (v >= theta_max_deg) {
                o[x] = 255;
            }
            else {
                double t = v / theta_max_deg; // 0~1
                int pix = static_cast<int>(t * 255.0 + 0.5);
                o[x] = static_cast<unsigned char>(std::clamp(pix, 0, 255));
            }
        }
    }
    return out;
}

cv::Mat TerrainIO::VisualizeCost8U(const cv::Mat& cost64, const cv::Mat& obstacle8u, double inf_cost) {
    if (cost64.empty() || cost64.type() != CV_64FC1 || cost64.channels() != 1) {
        throw std::runtime_error("VisualizeCost8U: cost64 must be CV_64FC1 single-channel.");
    }
    if (obstacle8u.empty() || obstacle8u.type() != CV_8UC1 || obstacle8u.channels() != 1) {
        throw std::runtime_error("VisualizeCost8U: obstacle8u must be CV_8UC1 single-channel.");
    }
    if (cost64.size() != obstacle8u.size()) {
        throw std::runtime_error("VisualizeCost8U: size mismatch.");
    }

    cv::Mat out(cost64.rows, cost64.cols, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < cost64.rows; ++y) {
        const double* c = cost64.ptr<double>(y);
        const unsigned char* o = obstacle8u.ptr<unsigned char>(y);
        unsigned char* dst = out.ptr<unsigned char>(y);

        for (int x = 0; x < cost64.cols; ++x) {
            if (o[x] != 0 || c[x] >= inf_cost * 0.5) {
                dst[x] = 255; // 障碍
            }
            else {
                // 这里默认 cost 在 [0,1]（坡度代价策略）或 0（距离最短策略）
                double v = c[x];
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                int pix = static_cast<int>(v * 255.0 + 0.5);
                dst[x] = static_cast<unsigned char>(std::clamp(pix, 0, 255));
            }
        }
    }
    return out;
}

void TerrainIO::SaveGrayPng(const std::string& path, const cv::Mat& img8u) {
    if (img8u.empty() || img8u.type() != CV_8UC1) {
        throw std::runtime_error("SaveGrayPng: img must be CV_8UC1.");
    }
    namespace fs = std::filesystem;
    fs::path p(path);
    if (!p.parent_path().empty()) {
        fs::create_directories(p.parent_path());
    }
    if (!cv::imwrite(path, img8u)) {
        throw std::runtime_error("SaveGrayPng failed: " + path);
    }
}
