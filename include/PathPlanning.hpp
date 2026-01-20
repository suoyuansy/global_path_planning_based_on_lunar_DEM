#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

class PathPlanning {
public:
    explicit PathPlanning(const std::string& color_png_path);

private:
    //文件路径
    std::string color_png_;
    std::string costmap_dir_ = "out_put/costmap";
    std::string costmap_file_;

    cv::Mat base_img_;
    cv::Mat display_;

    cv::Mat costmap_;                    // CV_64FC1  1.0=障碍 存储原始代价图
    cv::Mat costmap_add_;                // 叠加的代价图 可以用costmap_还原


    cv::Point start_pt_;
    cv::Point goal_pt_;

    int  strategy_ = 0;     // 1 slope 2 rough 3 step 4 merge
    bool dist_first_ = false;
    bool expand_ = false;
    bool use_artificial_ = false;         // 是否启用人工代价

    // 人工点选择状态
    enum class State { SELECT_START = 0, SELECT_GOAL, SELECT_ARTI_TYPE, SELECT_ARTI_POINT };
    State current_state_ = State::SELECT_START;
    bool is_guide_next_ = true;  // 下一个待选人工点类型
    int arti_counter_ = 0;       // 已选人工点计数             
    std::vector<cv::Point> guide_pts_;      // 导引点序列
    std::vector<cv::Point> reject_pts_;     // 排斥点序列

    int radius = 100;  // 圆形影响范围半径（像素）


    void consoleInput_();
    void loadColorImage_();
    void searchCostmapFile_();
    void loadCostmap_();
    void showAndInteract_();
    
    
    void redraw_(const std::vector<cv::Point>& path = {});
    bool isObstacle_(int x, int y) const { return std::abs(costmap_.at<double>(y, x) - 1.0) < 1e-6; }
    std::vector<cv::Point> planAStar_() const;
    void savePath_(const std::vector<cv::Point>& path) const;
    static void mouseCallback_(int event, int x, int y, int flags, void* userdata);
    void onMouse_(int x, int y);

    // 人工代价相关函数
    void selectArtificialPointsLoop_();                    // 人工点选择主循环
    void addArtificialCost_(const cv::Point& center, bool is_guide);  // 添加单个人工代价
    cv::Mat computeArtificialCostmap_(const cv::Point& center, bool is_guide) const;  // 计算单个人工代价图
    double artificialCostFunction_(double dist, bool is_guide) const;  // 代价函数计算
    std::string generateArtificialSuffix_() const;       // 生成文件名后缀
};