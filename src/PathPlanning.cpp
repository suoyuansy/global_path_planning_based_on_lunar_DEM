#include "PathPlanning.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <iomanip>

namespace fs = std::filesystem;

/* ---------------- 构造  ---------------- */
PathPlanning::PathPlanning(const std::string& color_png_path)
    : color_png_(color_png_path) 
{
    consoleInput_();      // 用户输入策略
    loadColorImage_();    // 读底图
    searchCostmapFile_(); // 找代价文件
    loadCostmap_();       // 加载代价
    showAndInteract_();   // 交互主循环

}

/* ---------------- 控制台输入 ---------------- */
void PathPlanning::consoleInput_() {
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

    std::cout << "Press ESC at any time to exit\n";
}

/* ---------------- 读取底图 ---------------- */
void PathPlanning::loadColorImage_() {
    base_img_ = cv::imread(color_png_);
    if (base_img_.empty()) throw std::runtime_error("Cannot read data/color.png");
    display_ = base_img_.clone();
}

/* ---------------- 搜索代价文件 ---------------- */
void PathPlanning::searchCostmapFile_() {
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
        costmap_file_ = e.path().string();
        std::cout << "Costmap loaded: " << costmap_file_ << "\n";
        return;
    }
    throw std::runtime_error("Costmap file not found in out_put/costmap");
}

/* ---------------- 加载代价 ---------------- */
void PathPlanning::loadCostmap_() {
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

}

/* ---------------- 显示 + 交互 ---------------- */
void PathPlanning::showAndInteract_() 
{
    redraw_();
    cv::namedWindow("PathPlanning", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("PathPlanning", mouseCallback_, this);

    std::cout << "Please select start point (left click)\n";
    while (true) 
    {
        int key = cv::waitKey(30) & 0xFF;
        if (key == 27) { std::cout << "ESC exit\n"; break; }
    }
}

/* ---------------- 重绘画布 ---------------- */
void PathPlanning::redraw_() {
    display_ = base_img_.clone();
    for (int y = 0; y < costmap_.rows; ++y) {
        const double* p = costmap_.ptr<double>(y);
        for (int x = 0; x < costmap_.cols; ++x)
            if (std::abs(p[x] - 1.0) < 1e-6) display_.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
    }
    cv::imshow("PathPlanning", display_);
}



/* ---------------- 八方向 A* 规划（欧氏距离） ---------------- */
std::vector<cv::Point> PathPlanning::planAStar_() const {
    // 八方向增量
    const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    const int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const double step_cost[8] = { 1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0),
                                 1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0) }; // 直线 vs 对角线

    int w = costmap_.cols, h = costmap_.rows;

    cv::Mat visited(h, w, CV_8U, cv::Scalar(0));
    cv::Mat dist(h, w, CV_64FC1, cv::Scalar(1e10));
    cv::Mat parent_x(h, w, CV_32S, cv::Scalar(-1));
    cv::Mat parent_y(h, w, CV_32S, cv::Scalar(-1));

    using Node = std::tuple<double, int, int>; // f = g + h, x, y
    std::priority_queue<Node, std::vector<Node>, std::greater<>> open;
    open.emplace(0.0, start_pt_.x, start_pt_.y);
    dist.at<double>(start_pt_.y, start_pt_.x) = 0.0;

    while (!open.empty()) {
        auto [f, cx, cy] = open.top(); open.pop();
        if (visited.at<uchar>(cy, cx)) continue;
        visited.at<uchar>(cy, cx) = 1;
        if (cx == goal_pt_.x && cy == goal_pt_.y) break;

        for (int dir = 0; dir < 8; ++dir) {
            int nx = cx + dx[dir];
            int ny = cy + dy[dir];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            if (isObstacle_(nx, ny)) continue;
            if (visited.at<uchar>(ny, nx)) continue;

            double g = dist.at<double>(cy, cx) + 1000*step_cost[dir] * costmap_.at<double>(ny, nx);
            if (g < dist.at<double>(ny, nx)) {
                dist.at<double>(ny, nx) = g;
                parent_x.at<int>(ny, nx) = cx;
                parent_y.at<int>(ny, nx) = cy;

                // 欧氏启发
                double h = std::hypot(nx - goal_pt_.x, ny - goal_pt_.y);
                open.emplace(g + h, nx, ny);

            }
        }
    }

    /* 回溯路径 */
    std::vector<cv::Point> path;
    int cx = goal_pt_.x, cy = goal_pt_.y;
    if (parent_x.at<int>(cy, cx) == -1) return path; // unreachable
    while (cx != start_pt_.x || cy != start_pt_.y) {
        path.emplace_back(cx, cy);
        int px = parent_x.at<int>(cy, cx);
        int py = parent_y.at<int>(cy, cx);
        cx = px; cy = py;
    }
    path.emplace_back(start_pt_);
    std::reverse(path.begin(), path.end());
    return path;
}


/* ---------------- 保存路径 ---------------- */
void PathPlanning::savePath_(const std::vector<cv::Point>& path) const
{
    fs::create_directories("out_put/path_planning");
    std::string base = fs::path(costmap_file_).stem().string();
    std::string fname = "out_put/path_planning/" + base +"_start(" + std::to_string(start_pt_.x) + "," +std::to_string(start_pt_.y) + ")_goal(" +std::to_string(goal_pt_.x) + "," +std::to_string(goal_pt_.y) + ").txt";
    std::ofstream ofs(fname);
    if (!ofs) throw std::runtime_error("Cannot write path file");
    for (size_t i = 0; i < path.size(); ++i) {
        ofs << "(" << path[i].x << "," << path[i].y << ")";
        if (i + 1 < path.size()) ofs << "->";
    }
    std::cout << "Path saved to " << fname << "\n";
}

/* ---------------- 鼠标回调 ---------------- */
void PathPlanning::mouseCallback_(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* self = static_cast<PathPlanning*>(userdata);
    self->onMouse_(x, y);
}

void PathPlanning::onMouse_(int x, int y) {
    static bool picking_start = true;
    if (x < 0 || y < 0 || x >= costmap_.cols || y >= costmap_.rows) return;
    if (isObstacle_(x, y)) {
        std::cout << "Obstacle here: (" << x << "," << y << "), please re-select\n";
        return;
    }
    if (picking_start) {
        start_pt_ = cv::Point(x, y);
        std::cout << "Start point selected: (" << x << "," << y << ")\n";
        picking_start = false;
        redraw_();
        cv::circle(display_, start_pt_, 3, cv::Vec3b(0, 255, 0), -1);
        cv::imshow("PathPlanning", display_);
        std::cout << "Please select goal point\n";
    }
    else {
        goal_pt_ = cv::Point(x, y);
        std::cout << "Goal point selected: (" << x << "," << y << ")\n";
        cv::circle(display_, goal_pt_, 3, cv::Vec3b(255, 0, 0), -1);
        cv::imshow("PathPlanning", display_);
        picking_start = true;
        std::cout << "Running A* ...\n";
        auto path = planAStar_();
        if (path.empty()) {
            std::cout << "No path found between the two points\n";
        }
        else {
            for (const auto& pt : path)
                display_.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(255, 255, 255);
            cv::imshow("PathPlanning", display_);
            savePath_(path);
        }
        std::cout << "Please select start point (or press ESC to exit)\n";
    }
}