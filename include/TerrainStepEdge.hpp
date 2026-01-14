#pragma once
#include <opencv2/core.hpp>
#include <string>

class TerrainStepEdge {
public:
    TerrainStepEdge(const cv::Mat& dem_m,const std::string& root_out,double max_step_m = 0.4, double inf_cost = 1e10);

    /* ---- 只读成果 ---- */
    const cv::Mat& step_gradient() const { return step_gradient_; }
    const cv::Mat& cost_step()     const { return cost_step_; }
    const cv::Mat& step_obstacle() const { return step_obstacle_; }
    const cv::Mat& cost_distance() const { return cost_distance_; }

private:
    cv::Mat dem_m_;
    double max_step_, inf_cost_;
    std::string root_out_, out_txt_dir_, out_img_dir_;

    cv::Mat step_gradient_;   // Roberts 梯度幅值（m）
    cv::Mat cost_step_;       // 连续代价 0-1e10
    cv::Mat step_obstacle_;   // 0/1 障碍
    cv::Mat cost_distance_;   // 距离最短策略

    /* ---------- 内部流程 ---------- */
    void computeGradient_();    //  梯度
    void buildCost_();          //  代价 & 障碍
    void export_file();         //  导出

    /* ---------- 算法 & IO ---------- */
    static void    exportText_(const cv::Mat& m, const std::string& path, int precision = 3);
    static cv::Mat visualizePNG_(const cv::Mat& m, double max_cost);
    static void    savePng_(const std::string& path, const cv::Mat& img);
};