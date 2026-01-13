#pragma once
#include <opencv2/core.hpp>
#include <string>

class TerrainObstacleExpand {
public:
    TerrainObstacleExpand(const cv::Mat& slope_obs,const cv::Mat& rough_obs,const cv::Mat& step_obs,const std::string& root_out,double expand_mm = 2000.0, double grid_size_m = 1.0);  

    const cv::Mat& slope_expand()  const { return slope_expand_; }
    const cv::Mat& rough_expand()  const { return rough_expand_; }
    const cv::Mat& step_expand()   const { return step_expand_; }
    const cv::Mat& union_mask()    const { return union_mask_; }
    const cv::Mat& union_expand()  const { return union_expand_; }

private:
    cv::Mat slope_obs_, rough_obs_, step_obs_;
    double expand_mm_, grid_m_;
    std::string root_out_, out_txt_dir_, out_img_dir_;

    cv::Mat slope_expand_;
    cv::Mat rough_expand_;
    cv::Mat step_expand_;
    cv::Mat union_mask_;
    cv::Mat union_expand_;

    void expandSingle_(const cv::Mat& src, cv::Mat& dst, int k) const;
    void export_file_();
    static void exportText_(const cv::Mat& m, const std::string& path);
    static cv::Mat visualizeObs8U_(const cv::Mat& obs);
    static void savePng_(const std::string& path, const cv::Mat& img);
};