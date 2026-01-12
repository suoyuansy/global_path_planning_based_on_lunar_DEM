#pragma once
#include <opencv2/core.hpp>
#include <string>

class TerrainSlopeAspect {
public:
    TerrainSlopeAspect(const cv::Mat& dem_m,const std::string& root_out,double grid_size = 1.0,double theta_max = 20.0,double inf_cost = 1e10);

    const cv::Mat& slope_deg()     const { return slope_deg_; }
    const cv::Mat& aspect_deg()    const { return aspect_deg_; }
    const cv::Mat& cost_distance() const { return cost_distance_; }
    const cv::Mat& cost_slope()    const { return cost_slope_; }
    const cv::Mat& obstacle()      const { return obstacle_; }

private:
    cv::Mat dem_m_;
    double grid_size_, theta_max_, inf_cost_;
    std::string root_out_, out_txt_dir_, out_img_dir_;

    cv::Mat slope_deg_;
    cv::Mat aspect_deg_;
    cv::Mat cost_distance_;
    cv::Mat cost_slope_;
    cv::Mat obstacle_;         

    void computeSlopeAspect_();
    void buildCost_();
    void export_file();

    /* ---------- ¹¤¾ßº¯Êý ---------- */
    static double rad2Deg_(double r);
    static double wrap360_(double a);
    static void   exportText_(const cv::Mat& m, const std::string& path, int precision = 3);
    static cv::Mat visualizePNG_(const cv::Mat& m, double theta_max);
    static void    savePng_(const std::string& path, const cv::Mat& img);

    static constexpr double kPi = 3.14159265358979323846;
};