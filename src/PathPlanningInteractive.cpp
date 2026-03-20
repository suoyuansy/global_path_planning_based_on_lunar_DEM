#include "PathPlanningInteractive.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "PathPlanner.hpp"

namespace fs = std::filesystem;

/* ---------------- 构造：旧模式 ---------------- */
PathPlanningInteractive::PathPlanningInteractive(const std::string& color_png_path)
    : color_png_(color_png_path),
    external_fixed_mode_(false),
    overwrite_path_mode_(false),
    keyboard_driven_mode_(false) {
    consoleInput_();
    loadColorImage_();
    searchCostmapFile_();
    loadCostmap_();
    showAndInteract_();
}

/* ---------------- 构造：新模式 ---------------- */
PathPlanningInteractive::PathPlanningInteractive(const std::string& color_png_path,
    const std::string& costmap_txt_path,
    const std::string& output_dir,
    bool enable_artificial)
    : color_png_(color_png_path),
    costmap_file_(costmap_txt_path),
    output_dir_(output_dir),
    external_fixed_mode_(true),
    overwrite_path_mode_(true),
    keyboard_driven_mode_(true),
    use_artificial_(enable_artificial) {
    loadColorImage_();
    loadCostmap_();
    showAndInteract_();
}

/* ---------------- 控制台输入（旧模式） ---------------- */
void PathPlanningInteractive::consoleInput_() {
    std::cout << "Select strategy (1 slope 2 roughness 3 step 4 merge): " << std::flush;
    std::cin >> strategy_;
    if (strategy_ < 1 || strategy_ > 4) {
        throw std::runtime_error("Invalid strategy");
    }

    int tmp = 0;
    std::cout << "Distance first? (0 no 1 yes): " << std::flush;
    std::cin >> tmp;
    if (tmp != 0 && tmp != 1) {
        throw std::runtime_error("Invalid distance-first flag");
    }
    dist_first_ = static_cast<bool>(tmp);

    std::cout << "Use expand? (0 no 1 yes): " << std::flush;
    std::cin >> tmp;
    if (tmp != 0 && tmp != 1) {
        throw std::runtime_error("Invalid expand flag");
    }
    expand_ = static_cast<bool>(tmp);

    std::cout << "Use artificial guidance/rejection? (0 no 1 yes): " << std::flush;
    std::cin >> tmp;
    if (tmp != 0 && tmp != 1) {
        throw std::runtime_error("Invalid artificial flag");
    }
    use_artificial_ = static_cast<bool>(tmp);

    std::cout << "Press ESC at any time to exit\n" << std::flush;
}

/* ---------------- 读取底图 ---------------- */
void PathPlanningInteractive::loadColorImage_() {
    base_img_ = cv::imread(color_png_);
    if (base_img_.empty()) {
        throw std::runtime_error("Cannot read color image: " + color_png_);
    }
    display_ = base_img_.clone();
}

/* ---------------- 搜索代价文件（旧模式） ---------------- */
void PathPlanningInteractive::searchCostmapFile_() {
    std::string keyword;
    switch (strategy_) {
    case 1: keyword = "TerrainSlope"; break;
    case 2: keyword = "TerrainRoughness"; break;
    case 3: keyword = "TerrainStepEdge"; break;
    case 4: keyword = "Merge"; break;
    default: throw std::runtime_error("Invalid strategy");
    }

    for (const auto& e : fs::directory_iterator(costmap_dir_)) {
        std::string name = e.path().filename().string();
        if (name.find(keyword) == std::string::npos) continue;
        if (dist_first_ && name.find("distance") == std::string::npos) continue;
        if (!dist_first_ && name.find("distance") != std::string::npos) continue;
        if (expand_ && name.find("expand") == std::string::npos) continue;
        if (!expand_ && name.find("expand") != std::string::npos) continue;

        std::string unixPath = e.path().string();
        std::replace(unixPath.begin(), unixPath.end(), '\\', '/');
        costmap_file_ = unixPath;

        std::cout << "Costmap loaded: " << costmap_file_ << "\n" << std::flush;
        return;
    }

    throw std::runtime_error("Costmap file not found in out_put/costmap");
}

/* ---------------- 加载代价 ---------------- */
void PathPlanningInteractive::loadCostmap_() {
    if (costmap_file_.empty()) {
        throw std::runtime_error("Costmap file path is empty.");
    }

    std::ifstream ifs(costmap_file_);
    if (!ifs) {
        throw std::runtime_error("Cannot read costmap file: " + costmap_file_);
    }

    std::vector<std::vector<double>> tmp;
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double v;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) tmp.push_back(row);
    }

    if (tmp.empty()) {
        throw std::runtime_error("Costmap file is empty: " + costmap_file_);
    }

    int h = static_cast<int>(tmp.size());
    int w = static_cast<int>(tmp[0].size());
    costmap_.create(h, w, CV_64FC1);

    for (int y = 0; y < h; ++y) {
        if (static_cast<int>(tmp[y].size()) != w) {
            throw std::runtime_error("Costmap txt file is not rectangular.");
        }
        double* p = costmap_.ptr<double>(y);
        for (int x = 0; x < w; ++x) {
            p[x] = tmp[y][x];
        }
    }

    if (costmap_.size() != base_img_.size()) {
        throw std::runtime_error("Costmap size does not match base image size.");
    }

    costmap_add_ = costmap_.clone();
    std::cout << "Costmap loaded: " << costmap_file_ << "\n" << std::flush;
}

/* ---------------- 新模式：操作说明 ---------------- */
void PathPlanningInteractive::printInteractionHelp_() const {
    std::cout
        << "\n=== Interactive Global Path Planning ===\n"
        << "[Mouse]\n"
        << "  Left click #1 : select start point\n"
        << "  Left click #2 : select goal point\n"
        << "  GUIDE mode     : left click adds a guide point and replans\n"
        << "  REJECT mode    : left click adds a reject point and replans\n"
        << "\n[Keyboard]\n"
        << "  g : guide mode\n"
        << "  r : reject mode\n"
        << "  c : clear artificial points and replan\n"
        << "  f : finish current round and restart\n"
        << "  Esc : exit\n"
        << "========================================\n"
        << std::flush;
}

/* ---------------- 输出当前状态提示 ---------------- */
void PathPlanningInteractive::printCurrentStateHint_() const {
    switch (current_state_) {
    case State::SELECT_START:
        std::cout << "[State] Please left click to select START point.\n" << std::flush;
        break;
    case State::SELECT_GOAL:
        std::cout << "[State] Please left click to select GOAL point.\n" << std::flush;
        break;
    case State::SELECT_ARTI_TYPE:
        std::cout << "[State] Waiting for console input: 1=guide 2=reject 3=finish\n" << std::flush;
        break;
    case State::SELECT_ARTI_POINT:
        if (current_artificial_mode_ == ArtificialMode::GUIDE) {
            std::cout << "[State] GUIDE mode: left click to add a guide point.\n" << std::flush;
        }
        else if (current_artificial_mode_ == ArtificialMode::REJECT) {
            std::cout << "[State] REJECT mode: left click to add a reject point.\n" << std::flush;
        }
        else {
            std::cout << "[State] Path generated.\n" << std::flush;
        }
        break;
    }
}

/* ---------------- 显示 + 交互 ---------------- */
void PathPlanningInteractive::showAndInteract_() {
    redraw_();

    cv::namedWindow("PathPlanning", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("PathPlanning", mouseCallback_, this);

    if (keyboard_driven_mode_) {
        printInteractionHelp_();
    }
    else if (use_artificial_) {
        std::cout << "=== Artificial guidance/rejection mode enabled ===\n" << std::flush;
    }

    printCurrentStateHint_();

    while (true) {
        int key = cv::waitKey(30) & 0xFF;

        if (key == 27) {
            std::cout << "ESC exit\n" << std::flush;
            break;
        }

        if (!keyboard_driven_mode_) {
            continue;
        }

        if (key == 'g' || key == 'G') {
            if (!use_artificial_ || start_pt_.x < 0 || goal_pt_.x < 0) {
                continue;
            }
            current_artificial_mode_ = ArtificialMode::GUIDE;
            current_state_ = State::SELECT_ARTI_POINT;
            std::cout << "[Mode] GUIDE\n" << std::flush;
            printCurrentStateHint_();
        }
        else if (key == 'r' || key == 'R') {
            if (!use_artificial_ || start_pt_.x < 0 || goal_pt_.x < 0) {
                continue;
            }
            current_artificial_mode_ = ArtificialMode::REJECT;
            current_state_ = State::SELECT_ARTI_POINT;
            std::cout << "[Mode] REJECT\n" << std::flush;
            printCurrentStateHint_();
        }
        else if (key == 'c' || key == 'C') {
            if (start_pt_.x < 0 || goal_pt_.x < 0) {
                continue;
            }
            guide_pts_.clear();
            reject_pts_.clear();
            costmap_add_ = costmap_.clone();
            current_artificial_mode_ = ArtificialMode::NONE;
            current_state_ = State::SELECT_ARTI_POINT;
            std::cout << "[Action] Clear artificial points.\n" << std::flush;
            replanAndRefresh_();
            cv::waitKey(50);
            printCurrentStateHint_();
        }
        else if (key == 'f' || key == 'F') {
            std::cout << "[Action] Finish current round. Restarting...\n" << std::flush;

            guide_pts_.clear();
            reject_pts_.clear();
            costmap_add_ = costmap_.clone();
            start_pt_ = cv::Point(-1, -1);
            goal_pt_ = cv::Point(-1, -1);
            current_artificial_mode_ = ArtificialMode::NONE;
            current_state_ = State::SELECT_START;

            redraw_();
            cv::waitKey(50);
            printCurrentStateHint_();
        }
    }

    cv::destroyWindow("PathPlanning");
}

/* ---------------- 重绘画布：关键，最后立刻 imshow ---------------- */
void PathPlanningInteractive::redraw_(const std::vector<cv::Point>& path) {
    display_ = base_img_.clone();

    // 障碍物
    for (int y = 0; y < costmap_.rows; ++y) {
        const double* p = costmap_.ptr<double>(y);
        for (int x = 0; x < costmap_.cols; ++x) {
            if (std::abs(p[x] - 1.0) < 1e-6) {
                display_.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = 0.4;
    const int thickness = 1;
    const int offset = 8;

    // guide 点
    for (const auto& pt : guide_pts_) {
        cv::circle(display_, pt, 5, cv::Vec3b(0, 255, 255), -1);
        cv::circle(display_, pt, radius, cv::Vec3b(0, 255, 255), 1);
        cv::putText(display_, "guide",
            cv::Point(pt.x + offset, pt.y - offset),
            font, fontScale, cv::Vec3b(0, 255, 255), thickness);
    }

    // reject 点
    for (const auto& pt : reject_pts_) {
        cv::circle(display_, pt, 5, cv::Vec3b(255, 0, 255), -1);
        cv::circle(display_, pt, radius, cv::Vec3b(255, 0, 255), 1);
        cv::putText(display_, "reject",
            cv::Point(pt.x + offset, pt.y - offset),
            font, fontScale, cv::Vec3b(255, 0, 255), thickness);
    }

    // 起点
    if (start_pt_.x >= 0 && start_pt_.y >= 0) {
        cv::circle(display_, start_pt_, 4, cv::Vec3b(255, 0, 0), -1);
        cv::putText(display_, "start",
            cv::Point(start_pt_.x + offset, start_pt_.y - offset),
            font, fontScale, cv::Vec3b(255, 0, 0), thickness);
    }

    // 终点
    if (goal_pt_.x >= 0 && goal_pt_.y >= 0) {
        cv::circle(display_, goal_pt_, 4, cv::Vec3b(255, 255, 0), -1);
        cv::putText(display_, "goal",
            cv::Point(goal_pt_.x + offset, goal_pt_.y - offset),
            font, fontScale, cv::Vec3b(255, 255, 0), thickness);
    }

    // 路径
    for (const auto& pt : path) {
        if (pt != start_pt_ && pt != goal_pt_) {
            display_.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(255, 255, 255);
        }
    }

    // 关键：立刻刷新窗口
    cv::imshow("PathPlanning", display_);
}

/* ---------------- 统一重规划并刷新 ---------------- */
void PathPlanningInteractive::replanAndRefresh_() {
    if (start_pt_.x < 0 || goal_pt_.x < 0) {
        redraw_();
        return;
    }

    std::cout << "Running A* ...\n" << std::flush;
    auto path = PathPlanner::plan(planning_method_, costmap_add_, start_pt_, goal_pt_);

    if (path.empty()) {
        std::cout << "No path found!\n" << std::flush;
        redraw_();
        return;
    }

    redraw_(path);
    savePath_(path);
}

/* ---------------- 人工代价函数 ---------------- */
double PathPlanningInteractive::artificialCostFunction_(double dist, bool is_guide) const {
    const double K = 0.01;
    const double c = 5.0;

    if (is_guide) {
        if (dist > radius) return K * radius;
        return K * dist;
    }
    else {
        if (dist < 1e-6) return std::exp(-0.001 / (c * c));
        return std::exp(-0.001 * (dist * dist) / (c * c));
    }
}

/* ---------------- 计算单个人工代价图 ---------------- */
cv::Mat PathPlanningInteractive::computeArtificialCostmap_(const cv::Point& center, bool is_guide) const {
    cv::Mat artificial_cost = cv::Mat::zeros(costmap_.size(), CV_64FC1);

    int x_min = std::max(0, center.x - radius);
    int x_max = std::min(costmap_.cols - 1, center.x + radius);
    int y_min = std::max(0, center.y - radius);
    int y_max = std::min(costmap_.rows - 1, center.y + radius);

    if (is_guide) {
        for (int y = 0; y < costmap_.rows; ++y) {
            for (int x = 0; x < costmap_.cols; ++x) {
                if (!isObstacle_(x, y)) {
                    double dist = std::hypot(x - center.x, y - center.y);
                    artificial_cost.at<double>(y, x) = artificialCostFunction_(dist, true);
                }
            }
        }
    }
    else {
        for (int y = y_min; y <= y_max; ++y) {
            for (int x = x_min; x <= x_max; ++x) {
                double dist = std::hypot(x - center.x, y - center.y);
                if (dist <= radius && !isObstacle_(x, y)) {
                    artificial_cost.at<double>(y, x) = artificialCostFunction_(dist, false);
                }
            }
        }
    }

    return artificial_cost;
}

/* ---------------- 添加人工代价 ---------------- */
void PathPlanningInteractive::addArtificialCost_(const cv::Point& center, bool is_guide) {
    cv::Mat arti_map = computeArtificialCostmap_(center, is_guide);
    cv::add(costmap_add_, arti_map, costmap_add_);
}

/* ---------------- 文件名后缀 ---------------- */
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

/* ---------------- 保存路径 ---------------- */
void PathPlanningInteractive::savePath_(const std::vector<cv::Point>& path) const {
    if (overwrite_path_mode_) {
        fs::create_directories(output_dir_);
        std::string fname = (fs::path(output_dir_) / "path.txt").string();

        std::ofstream ofs(fname);
        if (!ofs) {
            throw std::runtime_error("Cannot write path file: " + fname);
        }

        for (size_t i = 0; i < path.size(); ++i) {
            ofs << "(" << path[i].x << "," << path[i].y << ")";
            if (i + 1 < path.size()) ofs << "->";
        }

        std::cout << "Path saved to " << fname << "\n" << std::flush;
        return;
    }

    fs::create_directories("out_put/path_planning");
    std::string base = fs::path(costmap_file_).stem().string();
    std::string fname = "out_put/path_planning/" + base
        + "_start(" + std::to_string(start_pt_.x) + "," + std::to_string(start_pt_.y) + ")"
        + generateArtificialSuffix_()
        + "_goal(" + std::to_string(goal_pt_.x) + "," + std::to_string(goal_pt_.y) + ").txt";

    std::ofstream ofs(fname);
    if (!ofs) {
        throw std::runtime_error("Cannot write path file");
    }

    for (size_t i = 0; i < path.size(); ++i) {
        ofs << "(" << path[i].x << "," << path[i].y << ")";
        if (i + 1 < path.size()) ofs << "->";
    }

    std::cout << "Path saved to " << fname << "\n" << std::flush;
}

/* ---------------- 旧模式：控制台人工点循环 ---------------- */
void PathPlanningInteractive::selectArtificialPointsLoop_() {
    std::cout << "\n=== Artificial point #" << (arti_counter_ + 1) << " ===\n";
    std::cout << "Select control point type (1=guide 2=reject 3=finish): " << std::flush;

    int choice;
    std::cin >> choice;

    if (choice == 3) {
        std::cout << "Finished adding artificial points. Restarting...\n" << std::flush;

        guide_pts_.clear();
        reject_pts_.clear();
        goal_pt_ = cv::Point(-1, -1);
        start_pt_ = cv::Point(-1, -1);
        costmap_add_ = costmap_.clone();
        arti_counter_ = 0;

        current_artificial_mode_ = ArtificialMode::NONE;
        current_state_ = State::SELECT_START;

        redraw_();
        cv::waitKey(50);

        std::cout << "Please select start point\n" << std::flush;
        return;
    }

    if (choice != 1 && choice != 2) {
        std::cout << "Invalid input. Please enter 1, 2 or 3\n" << std::flush;
        selectArtificialPointsLoop_();
        return;
    }

    is_guide_next_ = (choice == 1);
    current_state_ = State::SELECT_ARTI_POINT;

    if (is_guide_next_) {
        std::cout << "guide point selected. Please click on the map.\n" << std::flush;
    }
    else {
        std::cout << "reject point selected. Please click on the map.\n" << std::flush;
    }
}

/* ---------------- 鼠标回调 ---------------- */
void PathPlanningInteractive::mouseCallback_(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* self = static_cast<PathPlanningInteractive*>(userdata);
    self->onMouse_(x, y);
}

/* ---------------- 鼠标逻辑 ---------------- */
void PathPlanningInteractive::onMouse_(int x, int y) {
    if (x < 0 || y < 0 || x >= costmap_.cols || y >= costmap_.rows) {
        return;
    }

    /* ---------- 选择起点 ---------- */
    if (current_state_ == State::SELECT_START) {
        if (isObstacle_(x, y)) {
            std::cout << "Obstacle at (" << x << "," << y << "), please re-select\n" << std::flush;
            return;
        }

        start_pt_ = cv::Point(x, y);
        goal_pt_ = cv::Point(-1, -1);

        guide_pts_.clear();
        reject_pts_.clear();
        costmap_add_ = costmap_.clone();
        arti_counter_ = 0;
        current_artificial_mode_ = ArtificialMode::NONE;

        std::cout << "[Point] Start selected: (" << x << "," << y << ")\n" << std::flush;

        current_state_ = State::SELECT_GOAL;
        redraw_();
        cv::waitKey(50);
        printCurrentStateHint_();
        return;
    }

    /* ---------- 选择终点 ---------- */
    if (current_state_ == State::SELECT_GOAL) {
        if (isObstacle_(x, y)) {
            std::cout << "Obstacle at (" << x << "," << y << "), please re-select\n" << std::flush;
            return;
        }

        goal_pt_ = cv::Point(x, y);
        std::cout << "[Point] Goal selected: (" << x << "," << y << ")\n" << std::flush;

        // 先把终点立即画出来
        redraw_();
        cv::waitKey(50);

        std::cout << "Running A* ...\n" << std::flush;
        auto path = PathPlanner::plan(planning_method_, costmap_add_, start_pt_, goal_pt_);

        if (path.empty()) {
            std::cout << "No path found!\n" << std::flush;
            current_state_ = State::SELECT_START;
            redraw_();
            cv::waitKey(50);
            std::cout << "Please select start point\n" << std::flush;
            return;
        }

        redraw_(path);
        savePath_(path);
        cv::waitKey(50);

        if (keyboard_driven_mode_) {
            current_state_ = State::SELECT_ARTI_POINT;
            current_artificial_mode_ = ArtificialMode::NONE;
            printCurrentStateHint_();
        }
        else {
            if (use_artificial_) {
                current_state_ = State::SELECT_ARTI_TYPE;
                selectArtificialPointsLoop_();
            }
            else {
                current_state_ = State::SELECT_START;
                std::cout << "Please select start point\n" << std::flush;
            }
        }
        return;
    }

    /* ---------- 选择人工点 ---------- */
    if (current_state_ == State::SELECT_ARTI_POINT) {
        /* ===== 新模式：键盘驱动 ===== */
        if (keyboard_driven_mode_) {
            if (!use_artificial_) {
                return;
            }

            if (current_artificial_mode_ == ArtificialMode::NONE) {
                std::cout << "[Info] Please press g or r first.\n" << std::flush;
                return;
            }

            if (isObstacle_(x, y)) {
                std::cout << "[Warning] Selected obstacle area for artificial point: ("
                    << x << "," << y << ")\n" << std::flush;
            }

            if (current_artificial_mode_ == ArtificialMode::GUIDE) {
                guide_pts_.push_back(cv::Point(x, y));
                std::cout << "[Point] Guide point added: (" << x << "," << y << ")\n" << std::flush;
                addArtificialCost_(cv::Point(x, y), true);
            }
            else {
                reject_pts_.push_back(cv::Point(x, y));
                std::cout << "[Point] Reject point added: (" << x << "," << y << ")\n" << std::flush;
                addArtificialCost_(cv::Point(x, y), false);
            }

            replanAndRefresh_();
            cv::waitKey(50);   // 外部模式人工点后立刻刷新
            printCurrentStateHint_();
            return;
        }

        /* ===== 旧模式：控制台 guide/reject/finish ===== */
        if (isObstacle_(x, y)) {
            std::cout << "Warning: Selected obstacle area for artificial point\n" << std::flush;
        }

        if (is_guide_next_) {
            guide_pts_.push_back(cv::Point(x, y));
        }
        else {
            reject_pts_.push_back(cv::Point(x, y));
        }

        arti_counter_++;

        redraw_();
        cv::waitKey(50);

        if (is_guide_next_) {
            std::cout << "Artificial guide point selected: (" << x << "," << y << ")\n" << std::flush;
        }
        else {
            std::cout << "Artificial reject point selected: (" << x << "," << y << ")\n" << std::flush;
        }

        addArtificialCost_(cv::Point(x, y), is_guide_next_);

        std::cout << "Re-running A* with artificial cost...\n" << std::flush;
        auto path = PathPlanner::plan(planning_method_, costmap_add_, start_pt_, goal_pt_);

        if (path.empty()) {
            std::cout << "No path found with current artificial points!\n" << std::flush;
            redraw_();
            cv::waitKey(50);
        }
        else {
            redraw_(path);
            savePath_(path);
            cv::waitKey(50);   // 旧模式人工点后立刻刷新
        }

        current_state_ = State::SELECT_ARTI_TYPE;
        selectArtificialPointsLoop_();
    }
}