#include "TerrainRoughness.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cmath>

/* ---------------- 对外入口 ---------------- */
TerrainRoughness::TerrainRoughness(const cv::Mat& dem_m, const std::string& root_out,double grid_size,double Lv_max,double inf_cost)
  : dem_m_(dem_m), g_(grid_size), Lv_max_(Lv_max), inf_cost_(inf_cost), root_out_(root_out)
{
    namespace fs = std::filesystem;

    /* 根目录下创建 TerrainRoughness */
    const std::string root = root_out + "/TerrainRoughness";
    fs::create_directories(root);

    /* 创建两级子目录 */
    out_txt_dir_ = root + "/out_txt_file";
    out_img_dir_ = root + "/out_image_file";
    fs::create_directories(out_txt_dir_);
    fs::create_directories(out_img_dir_);

    computeRoughness_();          // 计算粗糙度 + 障碍
    buildCost_();                 // 两种代价
    export_file();                // 输出

}

/* ---------- 3×3 平面拟合：计算粗糙度 & 障碍 ---------- */
void TerrainRoughness::computeRoughness_()
{
     const int rows = dem_m_.rows, cols = dem_m_.cols;
     roughness_.create(rows, cols, CV_64FC1);
     obstacle_.create(rows, cols, CV_8UC1);
     roughness_ = cv::Scalar(Lv_max_);   // 随后再赋值
     obstacle_ = cv::Scalar(1);

     if (rows < 3 || cols < 3) return;

     for (int y = 1; y < rows - 1; ++y) {
         for (int x = 1; x < cols - 1; ++x) {
             double Z[9] = {
                 dem_m_.at<double>(y - 1,x - 1), dem_m_.at<double>(y - 1,x), dem_m_.at<double>(y - 1,x + 1),
                 dem_m_.at<double>(y  ,x - 1), dem_m_.at<double>(y  ,x), dem_m_.at<double>(y  ,x + 1),
                 dem_m_.at<double>(y + 1,x - 1), dem_m_.at<double>(y + 1,x), dem_m_.at<double>(y + 1,x + 1)
             };
             double Lv = fitPlaneDistance_(Z, g_);
             roughness_.at<double>(y, x) = Lv;
             obstacle_.at<unsigned char>(y, x) = (Lv >= Lv_max_) ? 1 : 0;
         }
     }
}

/* ---------- 两种代价 ---------- */
void TerrainRoughness::buildCost_()
{
    const int rows = roughness_.rows;
    const int cols = roughness_.cols;

    cost_dist_.create(rows, cols, CV_64FC1);
    cost_roll_.create(rows, cols, CV_64FC1);

    for (int y = 0; y < rows; ++y) {
        const double* r = roughness_.ptr<double>(y);
        double* d = cost_dist_.ptr<double>(y);
        double* s = cost_roll_.ptr<double>(y);
        for (int x = 0; x < cols; ++x) {
            double Lv = r[x];
            if (Lv >= Lv_max_) {              // 统一障碍区
                d[x] = inf_cost_;
                s[x] = inf_cost_;
            }
            else {                            // 可通行区
                d[x] = 0.0;                     // 距离最短策略
                s[x] = Lv / Lv_max_;            // 侧翻代价最小策略
            }
        }
    }
}

/* ---------- 导出  ---------- */
void TerrainRoughness::export_file() 
{
    std::cout << "\nOutputs:\n";
    exportText_(roughness_, out_txt_dir_ + "/roughness_lv.txt", 3);
    std::cout << "  " << out_txt_dir_ << "/roughness_lv.txt\n";
    exportText_(cost_dist_, out_txt_dir_ + "/cost_rough_distance.txt", 3);
    std::cout << "  " << out_txt_dir_ << "/cost_rough_distance.txt\n";
    exportText_(cost_roll_, out_txt_dir_ + "/cost_rough_rollover.txt", 3);
    std::cout << "  " << out_txt_dir_ << "/cost_rough_rollover.txt\n";
    exportText_(obstacle_, out_txt_dir_ + "/obstacle_rough.txt");
    std::cout << "  " << out_txt_dir_ << "/obstacle_rough.txt\n";

    savePng_(out_img_dir_ + "/roughness_lv.png", visualizePNG_(roughness_, Lv_max_));
    std::cout << "  " << out_img_dir_ << "/roughness_lv.png\n";
    savePng_(out_img_dir_ + "/obstacle_rough.png", visualizePNG_(obstacle_, Lv_max_));
    std::cout << "  " << out_img_dir_ << "/obstacle_rough.png\n";
}




/* ---------- 平面拟合 ---------- */
double TerrainRoughness::fitPlaneDistance_(const double Z[9], double g)
{
    /* 同之前实现，返回 Lv 即可 */
    static const double dx[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    static const double dy[9] = { -1,-1,-1, 0, 0, 0, 1, 1, 1 };
    double sx = 0, sy = 0, sz = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0;
    for (int i = 0; i < 9; ++i) {
        double x = dx[i] * g, y = dy[i] * g, z = Z[i];
        sx += x; sy += y; sz += z;
        sxx += x * x; syy += y * y; sxy += x * y;
        sxz += x * z; syz += y * z;
    }
    double n = 9.0;
    double det = sxx * syy - sxy * sxy;
    if (std::fabs(det) < 1e-15) return 0.0;
    double a = (syy * sxz - sxy * syz) / det;
    double b = (sxx * syz - sxy * sxz) / det;
    double c = (sz - a * sx - b * sy) / n;

    double maxPos = 0.0, maxNeg = 0.0;
    for (int i = 0; i < 9; ++i) {
        double dist = Z[i] - (a * dx[i] * g + b * dy[i] * g + c);
        if (dist > maxPos) maxPos = dist;
        if (-dist > maxNeg) maxNeg = -dist;
    }
    return maxPos + maxNeg;
}


/* =================================================================== */
/* =                         IO 工具                                    = */
/* =================================================================== */

void TerrainRoughness::exportText_(const cv::Mat& m, const std::string& path, int precision)
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

    if (m.type() == CV_64FC1) {                          // 浮点矩阵
        ofs << std::fixed << std::setprecision(precision);
        for (int r = 0; r < rows; ++r) {
            const double* row = m.ptr<double>(r);
            for (int c = 0; c < cols; ++c) {
                ofs << row[c] << (c + 1 < cols ? ' ' : '\n');
            }
        }
    }
    else if (m.type() == CV_8UC1) {                    // 整型矩阵
        for (int r = 0; r < rows; ++r) {
            const unsigned char* row = m.ptr<unsigned char>(r);
            for (int c = 0; c < cols; ++c) {
                ofs << static_cast<int>(row[c]) << (c + 1 < cols ? ' ' : '\n');
            }
        }
    }
    else {
        throw std::runtime_error("exportText_: unsupported matrix type.");
    }
}


/* ---------- 可视化：粗糙度或障碍 → 8U 灰度图 ---------- */
cv::Mat TerrainRoughness::visualizePNG_(const cv::Mat& m, double Lv_max)
{
    if (m.empty() || m.channels() != 1)
        throw std::runtime_error("visualizePNG_: input must be single-channel CV_64FC1 or CV_8UC1.");

    const int rows = m.rows;
    const int cols = m.cols;
    cv::Mat out(rows, cols, CV_8UC1);

    if (m.type() == CV_64FC1) {                          // 粗糙度图
        const double  lvMax = Lv_max <= 0.0 ? 0.15 : Lv_max;
        for (int r = 0; r < rows; ++r) {
            const double* src = m.ptr<double>(r);
            unsigned char* dst = out.ptr<unsigned char>(r);
            for (int c = 0; c < cols; ++c) {
                double v = src[c];
                if (v < 0.0) v = 0.0;
                if (v >= lvMax) dst[c] = 255;
                else {
                    int pix = static_cast<int>((v / lvMax) * 255.0 + 0.5);
                    dst[c] = static_cast<unsigned char>(std::clamp(pix, 0, 255));
                }
            }
        }
    }
    else if (m.type() == CV_8UC1) {                    // 障碍图
        for (int r = 0; r < rows; ++r) {
            const unsigned char* src = m.ptr<unsigned char>(r);
            unsigned char* dst = out.ptr<unsigned char>(r);
            for (int c = 0; c < cols; ++c) {
                dst[c] = src[c] ? 255 : 0;
            }
        }
    }
    else {
        throw std::runtime_error("visualizePNG_: unsupported matrix type.");
    }
    return out;
}

void TerrainRoughness::savePng_(const std::string& path, const cv::Mat& img)
{
    namespace fs = std::filesystem;
    fs::path p(path);
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
    if (!cv::imwrite(path, img)) throw std::runtime_error("saveGrayPng_ failed: " + path);
}