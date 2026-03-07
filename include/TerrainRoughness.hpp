#pragma once
#include <opencv2/core.hpp>
#include <string>

class TerrainRoughness {

public:
    TerrainRoughness(const cv::Mat& dem_m, const std::string& root_out,double grid_size = 1.0,double Lv_max = 0.15,double inf_cost = 1e10, bool export_file_flag = true);
    /* ---- 只读成果 ---- */
    const cv::Mat& roughness()      const { return roughness_; }
    const cv::Mat& cost_distance()  const { return cost_dist_; }
    const cv::Mat& cost_rollover()  const { return cost_roll_; }
    const cv::Mat& obstacle()       const { return obstacle_; }

private:
    cv::Mat dem_m_;
    double g_, Lv_max_, inf_cost_;
    std::string root_out_, out_txt_dir_, out_img_dir_;

    cv::Mat roughness_;
    cv::Mat cost_dist_;
    cv::Mat cost_roll_;
    cv::Mat obstacle_;

    /* ---------- 内部流程 ---------- */
    void computeRoughness_();
    void buildCost_();
    void export_file();

    /* ---------- 算法 & IO ---------- */
    static double fitPlaneDistance_(const double Z[9], double g);
    static void     exportText_(const cv::Mat& m, const std::string& path, int precision = 3);
    static cv::Mat visualizePNG_(const cv::Mat& m, double Lv_max);
    static void     savePng_(const std::string& path, const cv::Mat& img);
};