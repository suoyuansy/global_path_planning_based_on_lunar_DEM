#include "PathPlanningInteractive.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <iomanip>
#include "PathPlanner.hpp"

namespace fs = std::filesystem;

/* ---------------- 构造  ---------------- */
PathPlanningInteractive::PathPlanningInteractive(const std::string& color_png_path)
    : color_png_(color_png_path) 
{
    consoleInput_();      // 用户输入策略
    loadColorImage_();    // 读底图
    searchCostmapFile_(); // 找代价文件
    loadCostmap_();       // 加载代价
    showAndInteract_();   // 交互主循环

}

/* ---------------- 控制台输入 ---------------- */
void PathPlanningInteractive::consoleInput_() {
    std::cout << "Select strategy (1 slope 2 roughness 3 step 4 merge): ";
    std::cin >> strategy_;
    if (strategy_ < 1 || strategy_ > 4) throw std::runtime_error("Invalid strategy");

    int tmp = 0;

    std::cout << "Distance first? (0 no 1 yes): ";
    std::cin >> tmp;
    if (tmp != 0 && tmp != 1) throw std::runtime_error("Invalid distance-first flag (must be 0 or 1)");
    dist_first_ = static_cast<bool>(tmp);

    std::cout << "Use expand? (0 no 1 yes): ";
    std::cin >> tmp;
    if (tmp != 0 && tmp != 1) throw std::runtime_error("Invalid expand flag (must be 0 or 1)");
    expand_ = static_cast<bool>(tmp);

    std::cout << "Use artificial guidance/rejection? (0 no 1 yes): ";
    std::cin >> tmp;
    if (tmp != 0 && tmp != 1) throw std::runtime_error("Invalid artificial flag (must be 0 or 1)");
    use_artificial_ = static_cast<bool>(tmp);

    std::cout << "Press ESC at any time to exit\n";
}

/* ---------------- 读取底图 ---------------- */
void PathPlanningInteractive::loadColorImage_() {
    base_img_ = cv::imread(color_png_);
    if (base_img_.empty()) throw std::runtime_error("Cannot read data/color.png");
    display_ = base_img_.clone();
}

/* ---------------- 搜索代价文件 ---------------- */
void PathPlanningInteractive::searchCostmapFile_() {
    std::string keyword;
    switch (strategy_) {
    case 1: keyword = "TerrainSlope"; break;
    case 2: keyword = "TerrainRoughness"; break;
    case 3: keyword = "TerrainStepEdge"; break;
    case 4: keyword = "Merge"; break;
    }
    for (const auto& e : fs::directory_iterator(costmap_dir_)) {
        std::string name = e.path().filename().string();
        if (name.find(keyword) == std::string::npos) continue;
        if (dist_first_ && name.find("distance") == std::string::npos) continue;
        if (!dist_first_ && name.find("distance") != std::string::npos) continue;
        if (expand_ && name.find("expand") == std::string::npos) continue;
        if (!expand_ && name.find("expand") != std::string::npos) continue;

        // 把路径分隔符统一换成 '/'输出在控制台
        std::string unixPath = e.path().string();
        std::replace(unixPath.begin(), unixPath.end(), '\\', '/');
        costmap_file_ = unixPath;
        std::cout << "Costmap loaded: " << costmap_file_ << "\n";
        return;
    }
    throw std::runtime_error("Costmap file not found in out_put/costmap");
}

/* ---------------- 加载代价 ---------------- */
void PathPlanningInteractive::loadCostmap_() {
    std::ifstream ifs(costmap_file_);
    if (!ifs) throw std::runtime_error("Cannot read costmap file");

    std::vector<std::vector<double>> tmp;
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double v; while (ss >> v) row.push_back(v);
        if (!row.empty()) tmp.push_back(row);
    }
    int h = static_cast<int>(tmp.size());
    int w = static_cast<int>(tmp[0].size());
    costmap_.create(h, w, CV_64FC1);
    for (int y = 0; y < h; ++y) {
        double* p = costmap_.ptr<double>(y);
        for (int x = 0; x < w; ++x) p[x] = tmp[y][x];
    }
    /* **** 尺寸一致性检查 **** */
    if (costmap_.size() != base_img_.size())
        throw std::runtime_error("Costmap size does not match base image size - please check map dimensions");

    costmap_add_ = costmap_.clone();  // 初始化叠加代价图
}

/* ---------------- 显示 + 交互 ---------------- */
void PathPlanningInteractive::showAndInteract_()
{
    redraw_();
    cv::namedWindow("PathPlanning", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("PathPlanning", mouseCallback_, this);

    if (use_artificial_) {
        std::cout << "=== Artificial guidance/rejection mode enabled ===\n";
    }
    std::cout << "Please select start point (left click)\n";
    while (true) 
    {
        int key = cv::waitKey(30) & 0xFF;
        if (key == 27) { std::cout << "ESC exit\n"; break; }
    }
}

/* ---------------- 重绘画布 ---------------- */
void PathPlanningInteractive::redraw_(const std::vector<cv::Point>& path) {
    display_ = base_img_.clone();
    // 绘制障碍物（黑色）
    for (int y = 0; y < costmap_.rows; ++y) {
        const double* p = costmap_.ptr<double>(y);
        for (int x = 0; x < costmap_.cols; ++x)
            if (std::abs(p[x] - 1.0) < 1e-6) display_.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
    }

    //  人工点 + 文字标签
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = 0.4;
    const int thickness = 1;
    int offset = 8;   // 文字与圆点的像素间隔

    // 绘制所有人工点
    for (const auto& pt : guide_pts_) {
        cv::circle(display_, pt, 5, cv::Vec3b(0, 255, 255), -1);  
        cv::circle(display_, pt, radius, cv::Vec3b(0, 255, 255), 1); // 绿色圆圈
        cv::putText(display_, "guide", cv::Point(pt.x + offset, pt.y - offset),font, fontScale, cv::Vec3b(0, 255, 255), thickness);
    }
    for (const auto& pt : reject_pts_) {
        cv::circle(display_, pt, 5, cv::Vec3b(255, 0, 255), -1);  
        cv::circle(display_, pt, radius, cv::Vec3b(255, 0, 255), 1);
        cv::putText(display_, "reject", cv::Point(pt.x + offset, pt.y - offset),font, fontScale, cv::Vec3b(255, 0, 255), thickness);
    }

    // 绘制起点和终点
    if (start_pt_.x >= 0 && start_pt_.y >= 0) {
        cv::circle(display_, start_pt_, 4, cv::Vec3b(255, 0, 0), -1);  // 起点：蓝色
        cv::putText(display_, "start", cv::Point(start_pt_.x + offset, start_pt_.y - offset),font, fontScale, cv::Vec3b(255, 0, 0), thickness);
    }
    if (goal_pt_.x >= 0 && goal_pt_.y >= 0) {
        cv::circle(display_, goal_pt_, 4, cv::Vec3b(255, 255, 0), -1); // 终点：青色
        cv::putText(display_, "end", cv::Point(goal_pt_.x + offset, goal_pt_.y - offset),font, fontScale, cv::Vec3b(255, 255, 0), thickness);

    }

    // 绘制当前路径
    for (const auto& pt : path) {
        if (pt != start_pt_ && pt != goal_pt_) {  // 避免覆盖起点终点
            display_.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(255, 255, 255);  // 路径：白色
        }
    }
    cv::imshow("PathPlanning", display_);
}

/* ---------------- 人工代价函数计算 ---------------- */
double PathPlanningInteractive::artificialCostFunction_(double dist, bool is_guide) const {
    const double K = 0.01;  // 缩放系数
    const double c = 5.0;  // 高斯函数参数

    if (is_guide) {
        // 使用线性导引代价：Z = K * dist
         // 超出范围：恒定代价 = K * radius（保证连续性）
        if (dist > radius) {
            return K * radius;
        }
        // 半径范围内：线性代价 = K * dist
        return K * dist;

    }
    else {
        // 使用高斯排斥代价：Z = exp(-dist²/c²)
        // 转换为排斥效果：dist越小，代价越大
        if (dist < 1e-6) return exp(-0.001 / (c * c));  // 中心点最大代价
        return std::exp(-0.001*(dist * dist) / (c * c));
    }
}

/* ---------------- 计算单个人工代价图 ---------------- */
cv::Mat PathPlanningInteractive::computeArtificialCostmap_(const cv::Point& center, bool is_guide) const {
    cv::Mat artificial_cost = cv::Mat::zeros(costmap_.size(), CV_64FC1);

    // 计算安全的边界框，确保不越界
    int x_min = std::max(0, center.x - radius);
    int x_max = std::min(costmap_.cols - 1, center.x + radius);
    int y_min = std::max(0, center.y - radius);
    int y_max = std::min(costmap_.rows - 1, center.y + radius);

    if (is_guide)
    {
        // 在边界框内遍历，检查是否在圆形范围内
        for (int y = 0; y < costmap_.rows; ++y) {
            for (int x = 0; x < costmap_.cols; ++x) {
                if (!isObstacle_(x, y)) {
                    double dist = std::hypot(x - center.x, y - center.y);
                    artificial_cost.at<double>(y, x) = artificialCostFunction_(dist, is_guide);
                }
            }
        }

    }
    else
    {
        for (int y = y_min; y <= y_max; ++y) {
            for (int x = x_min; x <= x_max; ++x) {
                double dist = std::hypot(x - center.x, y - center.y);
                // 只在半径范围内且非障碍区域计算代价
                if (dist <= radius && !isObstacle_(x, y)) {
                    artificial_cost.at<double>(y, x) = artificialCostFunction_(dist, is_guide);
                }

            }
        }

    }
    return artificial_cost;
}

/* ---------------- 添加人工代价到costmap_add_ ---------------- */
void PathPlanningInteractive::addArtificialCost_(const cv::Point& center, bool is_guide) {
    cv::Mat arti_map = computeArtificialCostmap_(center, is_guide);
    cv::add(costmap_add_, arti_map, costmap_add_);
}

/* ---------------- 生成文件名后缀 ---------------- */
std::string PathPlanningInteractive::generateArtificialSuffix_() const {
    std::string suffix;
    for (const auto& pt : guide_pts_) {
        suffix += "_a(" + std::to_string(pt.x) + "," + std::to_string(pt.y) + ")";
    }
    for (const auto& pt : reject_pts_) {
        suffix += "_p(" + std::to_string(pt.x) + "," + std::to_string(pt.y) + ")";
    }
    return suffix;
}

/* ---------------- 保存路径（包含人工点信息） ---------------- */
void PathPlanningInteractive::savePath_(const std::vector<cv::Point>& path) const {
    fs::create_directories("out_put/path_planning");
    std::string base = fs::path(costmap_file_).stem().string();
    
    // 生成包含人工点信息的文件名
    std::string fname = "out_put/path_planning/" + base + 
                       "_start(" + std::to_string(start_pt_.x) + "," + std::to_string(start_pt_.y) + ")" +
                       generateArtificialSuffix_() +
                       "_goal(" + std::to_string(goal_pt_.x) + "," + std::to_string(goal_pt_.y) + ").txt";
    
    std::ofstream ofs(fname);
    if (!ofs) throw std::runtime_error("Cannot write path file");
    for (size_t i = 0; i < path.size(); ++i) {
        ofs << "(" << path[i].x << "," << path[i].y << ")";
        if (i + 1 < path.size()) ofs << "->";
    }
    std::cout << "Path saved to " << fname << "\n";
}

/* ---------------- 人工点选择主循环 ---------------- */
void PathPlanningInteractive::selectArtificialPointsLoop_() {

    std::cout << "\n=== Artificial point #" << (arti_counter_ + 1) << " ===\n";
    std::cout << "Select control point type (1=guide 2=reject 3=finish): ";

    int choice;
    std::cin >> choice;

    if (choice == 3) {
        std::cout << "Finished adding artificial points. Restarting...\n";
        // 重置状态
        guide_pts_.clear();
        reject_pts_.clear();
        goal_pt_ = cv::Point(-1, -1);
        start_pt_ = cv::Point(-1, -1);
        costmap_add_ = costmap_.clone();
        arti_counter_ = 0;
        current_state_ = State::SELECT_START;
        redraw_();
        cv::waitKey(50);
        std::cout << "Please select start point\n";
        return;
    }

    if (choice != 1 && choice != 2) {
        std::cout << "Invalid input. Please enter 1, 2 or 3\n";
        selectArtificialPointsLoop_();  // 递归重试
        return;
    }

    is_guide_next_ = (choice == 1);
    std::string type_str = is_guide_next_ ? "guide" : "reject";
    std::cout << type_str << " point selected. Please click on the map.\n";
    current_state_ = State::SELECT_ARTI_POINT;
}

/* ---------------- 鼠标回调 ---------------- */
void PathPlanningInteractive::mouseCallback_(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* self = static_cast<PathPlanningInteractive*>(userdata);
    self->onMouse_(x, y);
}

void PathPlanningInteractive::onMouse_(int x, int y) {
    if (x < 0 || y < 0 || x >= costmap_.cols || y >= costmap_.rows) return;

    // 选择起点
    if (current_state_ == State::SELECT_START) {
        if (isObstacle_(x, y)) {
            std::cout << "Obstacle at (" << x << "," << y << "), please re-select\n";
            return;
        }
        start_pt_ = cv::Point(x, y);
        goal_pt_ = cv::Point(-1, -1);  // 清空旧目标点
        guide_pts_.clear();                // 清除导引点
        reject_pts_.clear();               // 清除排斥点
        arti_counter_ = 0;                 // 重置计数器
        costmap_add_ = costmap_.clone();   // 重置叠加代价图

        std::cout << "Start point selected: (" << x << "," << y << ")\n";
        current_state_ = State::SELECT_GOAL;
        redraw_();  // 只显示起点，清除旧路径和人工点
        std::cout << "Please select goal point\n";
    }
    // 选择终点
    else if (current_state_ == State::SELECT_GOAL) {
        if (isObstacle_(x, y)) {
            std::cout << "Obstacle at (" << x << "," << y << "), please re-select\n";
            return;
        }
        goal_pt_ = cv::Point(x, y);
        std::cout << "Goal point selected: (" << x << "," << y << ")\n";
        redraw_();
        cv::waitKey(50);

        // 规划第一条路径
        std::cout << "Running A* ...\n";
        auto path = PathPlanner::plan(planning_method_, costmap_add_, start_pt_, goal_pt_);
        if (path.empty()) {
            std::cout << "No path found!\n";
            current_state_ = State::SELECT_START;
            redraw_();
            std::cout << "Please select start point\n";
            return;
        }

        // 只调用一次redraw_，正确显示路径
        redraw_(path);
        savePath_(path);

        // 刷新窗口，确保路径立即显示
        cv::waitKey(50);  // 给OpenCV时间刷新窗口

        if (use_artificial_) {
            // 进入人工点选择循环
            current_state_ = State::SELECT_ARTI_TYPE;
            selectArtificialPointsLoop_();
        }
        else {
            // 不使用人工代价，回到起点选择
            current_state_ = State::SELECT_START;
            std::cout << "Please select start point\n";
        }
    }
    // 选择人工点位置
    else if (current_state_ == State::SELECT_ARTI_POINT) {
         //人工点可以选择障碍物位置
         if (isObstacle_(x, y)) {
             std::cout << "Warning: Selected obstacle area for artificial point\n";
         }

        // 记录人工点
        if (is_guide_next_) {
            guide_pts_.push_back(cv::Point(x, y));
        }
        else {
            reject_pts_.push_back(cv::Point(x, y));
        }
        arti_counter_++;

        redraw_();
        cv::waitKey(50);

        std::string type_str = is_guide_next_ ? "guide" : "reject";
        std::cout << "Artificial " << type_str << " point selected: (" << x << "," << y << ")\n";

        // 添加人工代价
        addArtificialCost_(cv::Point(x, y), is_guide_next_);

        // 重新规划路径
        std::cout << "Re-running A* with artificial cost...\n";
        auto path = PathPlanner::plan(planning_method_,costmap_add_,start_pt_,goal_pt_);

        if (path.empty()) {
            std::cout << "No path found with current artificial points!\n";
        }
        else {
            redraw_(path);
            savePath_(path);

            // 强制刷新窗口，确保路径立即显示
            cv::waitKey(50);
        }

        // 继续选择下一个人工点
        current_state_ = State::SELECT_ARTI_TYPE;
        selectArtificialPointsLoop_();
    }
}