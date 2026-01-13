#include "TerrainStepEdge.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

/* ---------------- 对外入口 ---------------- */
TerrainStepEdge::TerrainStepEdge(const cv::Mat& dem_m,const std::string& root_out,double max_step_m)
 : dem_m_(dem_m), max_step_(max_step_m), root_out_(root_out)
{
    namespace fs = std::filesystem;
    const std::string root = root_out + "/TerrainStepEdge";
    fs::create_directories(root);
    out_txt_dir_ = root + "/out_txt_file";
    out_img_dir_ = root + "/out_image_file";
    fs::create_directories(out_txt_dir_);
    fs::create_directories(out_img_dir_);

    computeRoberts_();
    export_file_();
}

/* ---------- Roberts 边缘检测 + 阈值 ---------- */
void TerrainStepEdge::computeRoberts_()
{
    const int rows = dem_m_.rows;
    const int cols = dem_m_.cols;
    step_obstacle_.create(rows, cols, CV_8UC1);
    step_obstacle_ = cv::Scalar(1);          // 默认不可通行

    if (rows < 2 || cols < 2) return;

    for (int y = 1; y < rows - 1; ++y) {//不考虑图像边缘
        const double* r0 = dem_m_.ptr<double>(y);
        const double* r1 = dem_m_.ptr<double>(y + 1);
        unsigned char* dst = step_obstacle_.ptr<unsigned char>(y);
        for (int x = 1; x < cols - 1; ++x) {
            // Roberts 模板
            double gx = r0[x] - r1[x + 1];
            double gy = r0[x + 1] - r1[x];
            double grad = std::sqrt(gx * gx + gy * gy);
            dst[x] = (grad >= max_step_) ? 1 : 0;
        }
    }
}

/* ---------- 导出  ---------- */
void TerrainStepEdge::export_file_()
{
    exportText_(step_obstacle_, out_txt_dir_ + "/step_obstacle.txt");
    savePng_(out_img_dir_ + "/step_obstacle.png", visualizeObs8U_(step_obstacle_));

    std::cout << "\nOutputs:\n";
    std::cout << "  " << out_txt_dir_ << "/step_obstacle.txt\n";
    std::cout << "  " << out_img_dir_ << "/step_obstacle.png\n";
}

/* ---------- 工具函数 ---------- */
cv::Mat TerrainStepEdge::visualizeObs8U_(const cv::Mat& obs)
{
    cv::Mat out(obs.size(), CV_8UC1);
    for (int y = 0; y < obs.rows; ++y) {
        const unsigned char* s = obs.ptr<unsigned char>(y);
        unsigned char* d = out.ptr<unsigned char>(y);
        for (int x = 0; x < obs.cols; ++x) d[x] = s[x] ? 255 : 0;
    }
    return out;
}
void TerrainStepEdge::savePng_(const std::string& path, const cv::Mat& img)
{
    namespace fs = std::filesystem;
    fs::path p(path);
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
    if (!cv::imwrite(path, img)) throw std::runtime_error("savePng_ failed: " + path);
}
void TerrainStepEdge::exportText_(const cv::Mat& m, const std::string& path)
{
    if (m.empty() || m.type() != CV_8UC1 || m.channels() != 1)
        throw std::runtime_error("exportText_: input must be CV_8UC1 single-channel.");
    namespace fs = std::filesystem;
    fs::path p(path);
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
    std::ofstream ofs(path);
    if (!ofs.is_open()) throw std::runtime_error("exportText_: failed to open " + path);
    for (int r = 0; r < m.rows; ++r) {
        const unsigned char* row = m.ptr<unsigned char>(r);
        for (int c = 0; c < m.cols; ++c) ofs << static_cast<int>(row[c]) << (c + 1 < m.cols ? ' ' : '\n');
    }
}