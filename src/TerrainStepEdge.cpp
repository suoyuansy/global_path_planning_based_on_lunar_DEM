#include "TerrainStepEdge.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

/* ---------------- 对外入口 ---------------- */
TerrainStepEdge::TerrainStepEdge(const cv::Mat& dem_m,const std::string& root_out,double max_step_m, double inf_cost, bool export_file_flag)
 : dem_m_(dem_m), max_step_(max_step_m), root_out_(root_out), inf_cost_(inf_cost)
{
    computeGradient_();   // 梯度 + 障碍
    buildCost_();         // 代价矩阵
    if (export_file_flag)
    {
        namespace fs = std::filesystem;
        const std::string root = root_out + "/TerrainStepEdge";
        fs::create_directories(root);
        out_txt_dir_ = root + "/out_txt_file";
        out_img_dir_ = root + "/out_image_file";
        fs::create_directories(out_txt_dir_);
        fs::create_directories(out_img_dir_);
        export_file();        // 导出
    }
}

/* ---------- 梯度 + 障碍 ---------- */
void TerrainStepEdge::computeGradient_()
{
    const int rows = dem_m_.rows;
    const int cols = dem_m_.cols;
    step_gradient_.create(rows, cols, CV_64FC1);
    step_obstacle_.create(rows, cols, CV_8UC1);

    /* 边界默认最大梯度 + 障碍 */
    step_gradient_ = cv::Scalar(max_step_);
    step_obstacle_ = cv::Scalar(1);

    if (rows < 2 || cols < 2) return;

    for (int y = 1; y < rows - 1; ++y) {
        const double* r0 = dem_m_.ptr<double>(y);
        const double* r1 = dem_m_.ptr<double>(y + 1);
        double* g = step_gradient_.ptr<double>(y);
        unsigned char* o = step_obstacle_.ptr<unsigned char>(y);
        for (int x = 1; x < cols - 1; ++x) {
            double gx = r0[x] - r1[x + 1];
            double gy = r0[x + 1] - r1[x];
            double grad = std::sqrt(gx * gx + gy * gy);
            g[x] = grad;
            o[x] = (grad >= max_step_) ? 1 : 0;
        }
    }
}

/* ---------- 两种代价 ---------- */
void TerrainStepEdge::buildCost_()
{
    const int rows = step_gradient_.rows;
    const int cols = step_gradient_.cols;
    cost_step_.create(rows, cols, CV_64FC1);
    cost_distance_.create(rows, cols, CV_64FC1);


    for (int y = 0; y < rows ; ++y) {
        const double* g = step_gradient_.ptr<double>(y);
        double* c = cost_step_.ptr<double>(y);
        double* d = cost_distance_.ptr<double>(y);
        for (int x = 0; x < cols ; ++x) {
            double grad = g[x];
            if (grad >= max_step_) {
                c[x] = inf_cost_;              // 阶梯代价
                d[x] = inf_cost_;
            }
            else {
                c[x] = grad / max_step_;  // 连续代价 0-1
                d[x] = 0.0;
            }
        }
    }
}

/* ---------- 导出 ---------- */
void TerrainStepEdge::export_file()
{
    std::cout << "\nOutputs:\n";
    exportText_(step_gradient_, out_txt_dir_ + "/step_gradient.txt", 3);
    std::cout << "  " << out_txt_dir_ << "/step_gradient.txt\n";
    exportText_(cost_step_, out_txt_dir_ + "/cost_step.txt", 3);
    std::cout << "  " << out_txt_dir_ << "/cost_step.txt\n";
    exportText_(step_obstacle_, out_txt_dir_ + "/step_obstacle.txt");
    std::cout << "  " << out_txt_dir_ << "/step_obstacle.txt\n";
    exportText_(cost_distance_, out_txt_dir_ + "/cost_distance.txt", 3);
    std::cout << "  " << out_txt_dir_ << "/cost_distance.txt\n";

    savePng_(out_img_dir_ + "/step_gradient.png", visualizePNG_(step_gradient_, max_step_));
    std::cout << "  " << out_img_dir_ << "/step_gradient.png\n";
    savePng_(out_img_dir_ + "/step_obstacle.png", visualizePNG_(step_obstacle_, max_step_));
    std::cout << "  " << out_img_dir_ << "/step_obstacle.png\n";
}

/* ---------- 工具函数 ---------- */
void TerrainStepEdge::exportText_(const cv::Mat& m, const std::string& path, int precision)
{
    if (m.empty() || m.channels() != 1)
        throw std::runtime_error("exportText_: input must be single-channel CV_64FC1 or CV_8UC1.");
    namespace fs = std::filesystem;
    fs::path p(path);
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
    std::ofstream ofs(path);
    if (!ofs.is_open()) throw std::runtime_error("exportText_: failed to open " + path);
    const int rows = m.rows;
    const int cols = m.cols;
    if (m.type() == CV_64FC1) {
        ofs << std::fixed << std::setprecision(precision);
        for (int r = 0; r < rows; ++r) {
            const double* row = m.ptr<double>(r);
            for (int c = 0; c < cols; ++c) ofs << row[c] << (c + 1 < cols ? ' ' : '\n');
        }
    }
    else if (m.type() == CV_8UC1) {
        for (int r = 0; r < rows; ++r) {
            const unsigned char* row = m.ptr<unsigned char>(r);
            for (int c = 0; c < cols; ++c) ofs << static_cast<int>(row[c]) << (c + 1 < cols ? ' ' : '\n');
        }
    }
    else {
        throw std::runtime_error("exportText_: unsupported matrix type.");
    }
}
cv::Mat TerrainStepEdge::visualizePNG_(const cv::Mat& m, double max_cost)
{
    if (m.empty() || m.channels() != 1)
        throw std::runtime_error("visualizePNG_: input must be single-channel CV_64FC1 or CV_8UC1.");
    const int rows = m.rows;
    const int cols = m.cols;
    cv::Mat out(rows, cols, CV_8UC1);
    if (m.type() == CV_64FC1) {                      // 连续代价
        const double cMax = max_cost <= 0.0 ? 1.0 : max_cost;
        for (int r = 0; r < rows; ++r) {
            const double* src = m.ptr<double>(r);
            unsigned char* dst = out.ptr<unsigned char>(r);
            for (int c = 0; c < cols; ++c) {
                double v = src[c];
                if (v >= 1e10) { dst[c] = 255; }
                else {
                    if (v < 0.0) v = 0.0;
                    if (v > cMax) v = cMax;
                    int pix = static_cast<int>((v / cMax) * 255.0 + 0.5);
                    dst[c] = static_cast<unsigned char>(std::clamp(pix, 0, 255));
                }
            }
        }
    }
    else if (m.type() == CV_8UC1) {                // 障碍 0/1
        for (int r = 0; r < rows; ++r) {
            const unsigned char* src = m.ptr<unsigned char>(r);
            unsigned char* dst = out.ptr<unsigned char>(r);
            for (int c = 0; c < cols; ++c) dst[c] = src[c] ? 255 : 0;
        }
    }
    else {
        throw std::runtime_error("visualizePNG_: unsupported matrix type.");
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