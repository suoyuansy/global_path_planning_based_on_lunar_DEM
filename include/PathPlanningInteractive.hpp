#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "PathPlanner.hpp"

class PathPlanningInteractive {
public:
    // 旧模式
    explicit PathPlanningInteractive(const std::string& color_png_path);

    // 新模式：外部固定 costmap + 输出目录
    PathPlanningInteractive(const std::string& color_png_path,
        const std::string& costmap_txt_path,
        const std::string& output_dir,
        bool enable_artificial = true);

private:
    std::string color_png_;
    std::string costmap_dir_ = "out_put/costmap";
    std::string costmap_file_;
    std::string output_dir_;

    bool external_fixed_mode_ = false;
    bool overwrite_path_mode_ = false;
    bool keyboard_driven_mode_ = false;

    cv::Mat base_img_;
    cv::Mat display_;
    cv::Mat costmap_;
    cv::Mat costmap_add_;

    cv::Point start_pt_ = cv::Point(-1, -1);
    cv::Point goal_pt_ = cv::Point(-1, -1);

    int strategy_ = 0;
    bool dist_first_ = false;
    bool expand_ = false;
    bool use_artificial_ = false;

    enum class State {
        SELECT_START = 0,
        SELECT_GOAL,
        SELECT_ARTI_TYPE,   // 旧模式专用
        SELECT_ARTI_POINT   // 新模式 / 旧模式都可用
    };
    State current_state_ = State::SELECT_START;

    enum class ArtificialMode {
        NONE = 0,
        GUIDE,
        REJECT
    };
    ArtificialMode current_artificial_mode_ = ArtificialMode::NONE;

    bool is_guide_next_ = true;
    int arti_counter_ = 0;

    std::vector<cv::Point> guide_pts_;
    std::vector<cv::Point> reject_pts_;
    int radius = 100;

    PathPlanner::Method planning_method_ = PathPlanner::Method::AStar;

    void consoleInput_();
    void loadColorImage_();
    void searchCostmapFile_();
    void loadCostmap_();
    void showAndInteract_();
    void redraw_(const std::vector<cv::Point>& path = {});
    void replanAndRefresh_();

    bool isObstacle_(int x, int y) const {
        return std::abs(costmap_.at<double>(y, x) - 1.0) < 1e-6;
    }

    void savePath_(const std::vector<cv::Point>& path) const;

    static void mouseCallback_(int event, int x, int y, int flags, void* userdata);
    void onMouse_(int x, int y);

    // 旧模式专用：保留原始控制台人工点流程
    void selectArtificialPointsLoop_();

    void addArtificialCost_(const cv::Point& center, bool is_guide);
    cv::Mat computeArtificialCostmap_(const cv::Point& center, bool is_guide) const;
    double artificialCostFunction_(double dist, bool is_guide) const;
    std::string generateArtificialSuffix_() const;

    // 新模式提示
    void printInteractionHelp_() const;
    void printCurrentStateHint_() const;
};