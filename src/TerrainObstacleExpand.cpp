#include "TerrainObstacleExpand.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

/* ---------------- 对外入口 ---------------- */
TerrainObstacleExpand::TerrainObstacleExpand(const cv::Mat& slope_obs,const cv::Mat& rough_obs,const cv::Mat& step_obs,const std::string& root_out,double expand_mm,double grid_size_m, bool export_file_flag)
  : slope_obs_(slope_obs), rough_obs_(rough_obs), step_obs_(step_obs),expand_mm_(expand_mm), grid_m_(grid_size_m), root_out_(root_out)
{

    int k = static_cast<int>(std::round(expand_mm / 1000.0 / grid_size_m));
    if (k < 1) k = 1;

    expandSingle_(slope_obs_, slope_expand_, k);
    expandSingle_(rough_obs_, rough_expand_, k);
    expandSingle_(step_obs_, step_expand_, k);

    union_mask_ = slope_obs_ | rough_obs_ | step_obs_;          // 逻辑或
    expandSingle_(union_mask_, union_expand_, k);

    if (export_file_flag)
    {
        namespace fs = std::filesystem;
        const std::string root = root_out + "/TerrainObstacleExpand";
        fs::create_directories(root);
        out_txt_dir_ = root + "/out_txt_file";
        out_img_dir_ = root + "/out_image_file";
        fs::create_directories(out_txt_dir_);
        fs::create_directories(out_img_dir_);
        export_file_();
    }
}

/* ---------- 障碍膨胀（k×k 卷积） ---------- */
void TerrainObstacleExpand::expandSingle_(const cv::Mat& src,cv::Mat& dst,int k) const
{
    const int rows = src.rows;
    const int cols = src.cols;
    const int border = k / 2;

    /* 先在内域做 k×k 膨胀 */
    dst.create(src.size(), CV_8UC1);
    dst = cv::Scalar(0);                      

    //这里应该考虑border与边界范围的关系，分为边界内，border内以及边界与border之间三个范围进行考虑，还不完善
    for (int y = border; y < rows - border; ++y) {
        for (int x = border; x < cols - border; ++x) {
            if (src.at<unsigned char>(y, x) == 1) {
                dst.at<unsigned char>(y, x) = 1;
                continue;
            }
            bool hit = false;
            for (int dy = -border; dy <= border && !hit; ++dy)
                for (int dx = -border; dx <= border && !hit; ++dx)
                {
                    /* 只统计「非边界」且「值为 1」的像素 */
                    if (y + dy >= 1 && y + dy <rows - 1 && x + dx >= 1 && x + dx <cols - 1 &&src.at<unsigned char>(y + dy, x + dx) == 1)
                        hit = true;
                }
            dst.at<unsigned char>(y, x) = hit ? 1 : 0;
        }
    }

    /* 最后复制边框（不膨胀） */
    for (int y = 0; y < rows; ++y) {
        dst.at<unsigned char>(y, 0) = src.at<unsigned char>(y, 0);
        dst.at<unsigned char>(y, cols - 1) = src.at<unsigned char>(y, cols - 1);
    }
    for (int x = 0; x < cols; ++x) {
        dst.at<unsigned char>(0, x) = src.at<unsigned char>(0, x);
        dst.at<unsigned char>(rows - 1, x) = src.at<unsigned char>(rows - 1, x);
    }
}

/* ---------- 导出 ---------- */
void TerrainObstacleExpand::export_file_()
{
    std::cout << "\nOutputs:\n";
    exportText_(slope_expand_, out_txt_dir_ + "/slope_expand_obstacle.txt");
    std::cout << "  " << out_txt_dir_ << "/slope_expand_obstacle.txt\n";
    exportText_(rough_expand_, out_txt_dir_ + "/rough_expand_obstacle.txt");
    std::cout << "  " << out_txt_dir_ << "/rough_expand_obstacle.txt\n";
    exportText_(step_expand_, out_txt_dir_ + "/step_expand_obstacle.txt");
    std::cout << "  " << out_txt_dir_ << "/step_expand_obstacle.txt\n";
    exportText_(union_mask_, out_txt_dir_ + "/union_obstacle.txt");
    std::cout << "  " << out_txt_dir_ << "/union_obstacle.txt\n";
    exportText_(union_expand_, out_txt_dir_ + "/union_expand_obstacle.txt");
    std::cout << "  " << out_txt_dir_ << "/union_expand_obstacle.txt\n";

    savePng_(out_img_dir_ + "/slope_expand_obstacle.png", visualizeObs8U_(slope_expand_));
    std::cout << "  " << out_img_dir_ << "/slope_expand_obstacle.png\n";
    savePng_(out_img_dir_ + "/rough_expand_obstacle.png", visualizeObs8U_(rough_expand_));
    std::cout << "  " << out_img_dir_ << "/rough_expand_obstacle.png\n";
    savePng_(out_img_dir_ + "/step_expand_obstacle.png", visualizeObs8U_(step_expand_));
    std::cout << "  " << out_img_dir_ << "/step_expand_obstacle.png\n";
    savePng_(out_img_dir_ + "/union_obstacle.png", visualizeObs8U_(union_mask_));
    std::cout << "  " << out_img_dir_ << "/union_obstacle.png\n";
    savePng_(out_img_dir_ + "/union_expand_obstacle.png", visualizeObs8U_(union_expand_));
    std::cout << "  " << out_img_dir_ << "/union_expand_obstacle.png\n";
}

/* ---------- 工具函数 ---------- */
void TerrainObstacleExpand::exportText_(const cv::Mat& m, const std::string& path)
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
cv::Mat TerrainObstacleExpand::visualizeObs8U_(const cv::Mat& obs)
{
    cv::Mat out(obs.size(), CV_8UC1);
    for (int y = 0; y < obs.rows; ++y) {
        const unsigned char* s = obs.ptr<unsigned char>(y);
        unsigned char* d = out.ptr<unsigned char>(y);
        for (int x = 0; x < obs.cols; ++x) d[x] = s[x] ? 255 : 0;
    }
    return out;
}
void TerrainObstacleExpand::savePng_(const std::string& path, const cv::Mat& img)
{
    namespace fs = std::filesystem;
    fs::path p(path);
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
    if (!cv::imwrite(path, img)) throw std::runtime_error("savePng_ failed: " + path);
}