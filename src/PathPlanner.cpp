#include "PathPlanner.hpp"
#include <queue>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <set>
#include <unordered_set>
#include <iostream>
#include <steering_functions/reeds_shepp_state_space/reeds_shepp_state_space.hpp>

/* =========================
 * 通用小工具
 * ========================= */
namespace {
    constexpr double INF_COST = 1e18;
    constexpr double EPS = 1e-9;
    constexpr double PI = 3.14159265358979323846;

    inline double normalizeAngle(double a)
    {
        while (a > PI) a -= 2.0 * PI;
        while (a < -PI) a += 2.0 * PI;
        return a;
    }

    inline double deg2rad(double deg)
    {
        return deg * PI / 180.0;
    }

    inline bool lexicographicallyLess(double a1, double a2, double b1, double b2)
    {
        if (a1 < b1 - EPS) return true;
        if (a1 > b1 + EPS) return false;
        return a2 < b2 - EPS;
    }

    inline double octileDistance(int x1, int y1, int x2, int y2)
    {
        const double dx = std::abs(x1 - x2);
        const double dy = std::abs(y1 - y2);
        const double dmin = std::min(dx, dy);
        const double dmax = std::max(dx, dy);
        return (dmax - dmin) + std::sqrt(2.0) * dmin;
    }

    inline bool inRange(const cv::Mat& m, int x, int y)
    {
        return (x >= 0 && y >= 0 && x < m.cols && y < m.rows);
    }

    inline void validateCostmapOrThrow(const cv::Mat& costmap, const char* func_name)
    {
        if (costmap.empty() || costmap.type() != CV_64FC1) {
            throw std::runtime_error(std::string(func_name) + ": costmap must be CV_64FC1.");
        }
    }

    inline void validatePointOrThrow(const cv::Mat& costmap, const cv::Point& p, const char* name, const char* func_name)
    {
        if (!inRange(costmap, p.x, p.y)) {
            throw std::runtime_error(std::string(func_name) + ": " + name + " out of range.");
        }
    }
}


bool PathPlanner::isObstacle_(const cv::Mat& costmap, int x, int y)
{
    return std::abs(costmap.at<double>(y, x) - 1.0) < 1e-6;
}

bool PathPlanner::isSafe_(const cv::Mat& costmap, int x, int y, int safe_radius)
{
    const int w = costmap.cols;
    const int h = costmap.rows;
    /* 安全区检查：
     * 以当前节点为圆心（此处近似用方形邻域），
     * 若 safe_radius 范围内存在障碍，则认为该点不安全。
     * 越界同样视为不安全。
     */
    for (int dy = -safe_radius; dy <= safe_radius; ++dy) {
        for (int dx = -safe_radius; dx <= safe_radius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) return false;// 越界也算不安全
            if (isObstacle_(costmap, nx, ny)) return false;
        }
    }
    return true;
}

/* =========================
 * 统一入口
 * ========================= */

std::vector<cv::Point> PathPlanner::plan(
    Method method,
    const cv::Mat& costmap,
    const cv::Point& start,
    const cv::Point& goal
)
{
    switch (method) {
    case Method::AStar:
        return planAStar(costmap, start, goal);

    case Method::DStarLite:
        return planDStarLite(costmap, start, goal);

    case Method::HybridAStar:
        return planHybridAStar(costmap, start, goal);

    case Method::BidirectionalAStar: 
        return planBidirectionalAStar(costmap, start, goal);

    default:
        throw std::runtime_error("Unsupported planning method.");
    }
}

/* ---------------- 八方向 A* 规划（欧氏距离） ---------------- */
std::vector<cv::Point> PathPlanner::planAStar(const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal)
{
    if (costmap.empty() || costmap.type() != CV_64FC1)
        throw std::runtime_error("planAStar: costmap must be CV_64FC1.");

    const int w = costmap.cols;
    const int h = costmap.rows;

    if (start.x < 0 || start.y < 0 || start.x >= w || start.y >= h)
        throw std::runtime_error("planAStar: start out of range.");
    if (goal.x < 0 || goal.y < 0 || goal.x >= w || goal.y >= h)
        throw std::runtime_error("planAStar: goal out of range.");

    if (isObstacle_(costmap, start.x, start.y))
        throw std::runtime_error("planAStar: start is obstacle.");
    if (isObstacle_(costmap, goal.x, goal.y))
        throw std::runtime_error("planAStar: goal is obstacle.");
    // 八方向增量
    const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    const int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const double step_cost[8] = {
        1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0),
        1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0)
    };

    cv::Mat visited(h, w, CV_8U, cv::Scalar(0));
    cv::Mat dist(h, w, CV_64FC1, cv::Scalar(1e10));
    cv::Mat parent_x(h, w, CV_32S, cv::Scalar(-1));
    cv::Mat parent_y(h, w, CV_32S, cv::Scalar(-1));

    using Node = std::tuple<double, int, int>;
    std::priority_queue<Node, std::vector<Node>, std::greater<>> open;

    open.emplace(0.0, start.x, start.y);
    dist.at<double>(start.y, start.x) = 0.0;

    while (!open.empty()) {
        auto [f, cx, cy] = open.top();
        open.pop();

        if (visited.at<uchar>(cy, cx)) continue;
        visited.at<uchar>(cy, cx) = 1;

        if (cx == goal.x && cy == goal.y) break;

        for (int dir = 0; dir < 8; ++dir) {
            int nx = cx + dx[dir];
            int ny = cy + dy[dir];

            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            /***** 安全区筛选 *****/
            if (!isSafe_(costmap, nx, ny, 2)) continue;
            if (visited.at<uchar>(ny, nx)) continue;
            // 使用 costmap_add_ 计算代价
            double g = dist.at<double>(cy, cx)+ step_cost[dir]+ 1000.0 * costmap.at<double>(ny, nx);

            if (g < dist.at<double>(ny, nx)) {
                dist.at<double>(ny, nx) = g;
                parent_x.at<int>(ny, nx) = cx;
                parent_y.at<int>(ny, nx) = cy;
                // 欧氏启发
                double h_cost = std::hypot(nx - goal.x, ny - goal.y);
                open.emplace(g + h_cost, nx, ny);
            }
        }
    }
    /* 回溯路径 */
    std::vector<cv::Point> path;
    int cx = goal.x;
    int cy = goal.y;

    if (parent_x.at<int>(cy, cx) == -1)
        return path;

    while (cx != start.x || cy != start.y) {
        path.emplace_back(cx, cy);
        int px = parent_x.at<int>(cy, cx);
        int py = parent_y.at<int>(cy, cx);
        cx = px;
        cy = py;
    }

    path.emplace_back(start);
    std::reverse(path.begin(), path.end());
    return path;
}

/* =========================
 * D* Lite
 * ========================= */

std::vector<cv::Point> PathPlanner::planDStarLite(
    const cv::Mat& costmap,
    const cv::Point& start,
    const cv::Point& goal,
    int safe_radius,
    double cost_weight
) {

    const double INF = 1e9;
    const double EPS = 1e-6;

    int rows = costmap.rows;
    int cols = costmap.cols;

    // ========== 调试信息 #0: 输入参数检查 ==========
    std::cout << "[D* Lite] ========== 初始化 ==========" << std::endl;
    std::cout << "[D* Lite] 地图尺寸: " << cols << " x " << rows << std::endl;
    std::cout << "[D* Lite] 起点: (" << start.x << ", " << start.y << ")" << std::endl;
    std::cout << "[D* Lite] 终点: (" << goal.x << ", " << goal.y << ")" << std::endl;
    std::cout << "[D* Lite] 安全半径: " << safe_radius << ", 代价权重: " << cost_weight << std::endl;

    // 检查起点终点合法性
    auto inMap = [&](int x, int y) {
        return x >= 0 && x < cols && y >= 0 && y < rows;
        };

    if (!inMap(start.x, start.y)) {
        std::cerr << "[D* Lite] ERROR: 起点超出地图范围!" << std::endl;
        return {};
    }
    if (!inMap(goal.x, goal.y)) {
        std::cerr << "[D* Lite] ERROR: 终点超出地图范围!" << std::endl;
        return {};
    }

    // 检查起点终点是否是障碍物
    if (isObstacle_(costmap, start.x, start.y)) {
        std::cerr << "[D* Lite] ERROR: 起点是障碍物! cost=" << costmap.at<double>(start.y, start.x) << std::endl;
        return {};
    }
    if (isObstacle_(costmap, goal.x, goal.y)) {
        std::cerr << "[D* Lite] ERROR: 终点是障碍物! cost=" << costmap.at<double>(goal.y, goal.x) << std::endl;
        return {};
    }

    // 检查起点终点安全性
    if (!isSafe_(costmap, start.x, start.y, safe_radius)) {
        std::cerr << "[D* Lite] WARNING: 起点不安全(安全半径内存在障碍)" << std::endl;
        // 可以选择继续或返回，这里选择继续但发出警告
    }
    if (!isSafe_(costmap, goal.x, goal.y, safe_radius)) {
        std::cerr << "[D* Lite] WARNING: 终点不安全(安全半径内存在障碍)" << std::endl;
    }

    // 初始化所有节点的 g 和 rhs 为无穷大
    // g(s)   : 从节点 s 到终点（goal）的实际最短代价估计
    // rhs(s) : 基于后继节点的"前瞻"估计值
    // rhs(s) = min_{ s' ∈ Succ(s)} [ c(s, s') + g(s') ]
    cv::Mat g(rows, cols, CV_64F, cv::Scalar(INF));
    cv::Mat rhs(rows, cols, CV_64F, cv::Scalar(INF));

    // 8邻接方向
    std::vector<cv::Point> directions = {
        {1,0},{-1,0},{0,1},{0,-1},
        {1,1},{1,-1},{-1,1},{-1,-1}
    };

    // 获取邻居（8邻域）
    auto getNbrs = [&](int x, int y) {
        std::vector<cv::Point> nbrs;
        for (auto& d : directions) {
            int nx = x + d.x;
            int ny = y + d.y;
            if (inMap(nx, ny)) {
                nbrs.emplace_back(nx, ny);
            }
        }
        return nbrs;
        };

    // 启发函数（欧氏距离）
    auto heuristic = [&](int x1, int y1, int x2, int y2) {
        return std::hypot(x1 - x2, y1 - y2);
        };

    // 移动代价计算（从s到s'的代价）
    auto moveCost = [&](int x1, int y1, int x2, int y2) -> double {
        if (!inMap(x2, y2)) return INF;

        double base = std::hypot(x1 - x2, y1 - y2);  // 基础距离代价

        // 获取目标点的代价值
        double occ = 0.0;
        int type = costmap.type();
        if (type == CV_8U) {
            occ = costmap.at<uchar>(y2, x2) / 255.0;
        }
        else if (type == CV_32F) {
            occ = costmap.at<float>(y2, x2);
        }
        else if (type == CV_64F) {
            occ = costmap.at<double>(y2, x2);
        }
        else {
            occ = 0.0;
        }

        // 如果代价值接近1（障碍物），返回无穷大
        // 引导点区域的代价值被增加了 0.5~1.0,导致识别为障碍
        if (abs(occ-1.0) < 1e-6) {
            return INF;
        }

        return base + cost_weight * occ;
        };

    // 优先队列节点
    struct Node {
        double k1, k2;
        int x, y;
        bool operator<(const Node& other) const {
            // 注意：std::priority_queue 默认是大顶堆，所以需要反着定义
            if (std::abs(k1 - other.k1) > 1e-6) return k1 > other.k1;
            return k2 > other.k2;
        }
    };

    std::priority_queue<Node> open;

    // 计算Key值 [min(g, rhs) + h(start, s); min(g, rhs)]
    auto calcKey = [&](int x, int y) -> std::pair<double, double> {
        double val = std::min(g.at<double>(y, x), rhs.at<double>(y, x));
        double h_val = heuristic(start.x, start.y, x, y);
        return std::make_pair(val + h_val, val);
        };

    // 更新顶点（D* Lite核心）
    std::function<void(int, int)> updateVertex = [&](int x, int y) {
        if (!inMap(x, y)) return;

        // 如果不是目标点，计算rhs值
        if (!(x == goal.x && y == goal.y)) {
            double best = INF;
            cv::Point best_pred;

            // 获取所有后继（在D* Lite中，s的后继是s'，使得s可以移动到s'）
            // 注意：这里要找 min_{s' ∈ Succ(s)}(c(s, s') + g(s'))
            auto succs = getNbrs(x, y);  // s的后继

            for (auto& s_next : succs) {
                double c = moveCost(x, y, s_next.x, s_next.y);  // c(s, s_next)
                if (c >= INF) continue;  // 不可达

                double g_next = g.at<double>(s_next.y, s_next.x);
                double val = c + g_next;

                if (val < best - EPS) {
                    best = val;
                    best_pred = s_next;
                }
            }

            rhs.at<double>(y, x) = best;
        }

        // 如果g ≠ rhs，加入优先队列
        double g_val = g.at<double>(y, x);
        double rhs_val = rhs.at<double>(y, x);

        if (std::abs(g_val - rhs_val) > EPS) {
            auto [k1, k2] = calcKey(x, y);
            open.push(Node{ k1, k2, x, y });
        }
    };

    // ========== 调试信息 #1: 初始化完成 ==========
    std::cout << "[D* Lite] 初始化完成，开始设置目标点..." << std::endl;

    // 初始化：设置目标点
    rhs.at<double>(goal.y, goal.x) = 0.0;
    auto [k1_init, k2_init] = calcKey(goal.x, goal.y);
    open.push(Node{ k1_init, k2_init, goal.x, goal.y });

    std::cout << "[D* Lite] 目标点Key: [" << k1_init << ", " << k2_init << "]" << std::endl;

    // ========== 主循环 ==========
    int iteration = 0;
    const int MAX_ITERATIONS = rows * cols * 10;  // 防止无限循环

    std::cout << "[D* Lite] ========== 开始主循环 ==========" << std::endl;

    while (!open.empty()) {
        iteration++;

        // 防止无限循环
        if (iteration > MAX_ITERATIONS) {
            std::cerr << "[D* Lite] ERROR: 超过最大迭代次数 " << MAX_ITERATIONS << "，可能陷入死循环!" << std::endl;
            std::cerr << "[D* Lite] 当前队列大小: " << open.size() << std::endl;
            break;
        }

        // 每1000次迭代输出一次状态
        if (iteration % 1000 == 0) {
            std::cout << "[D* Lite] 迭代 " << iteration << ", 队列大小: " << open.size()
                << ", 当前起点g=" << g.at<double>(start.y, start.x)
                << ", rhs=" << rhs.at<double>(start.y, start.x) << std::endl;
        }

        Node u = open.top();
        open.pop();

        // 计算当前节点的最新Key值
        auto [k1_new, k2_new] = calcKey(u.x, u.y);

        // Key值过期检查（如果队列中的Key小于当前计算的Key，说明过时了，重新加入）
        if (u.k1 < k1_new - EPS || (std::abs(u.k1 - k1_new) < EPS && u.k2 < k2_new - EPS)) {
            open.push(Node{ k1_new, k2_new, u.x, u.y });
            continue;
        }

        double g_old = g.at<double>(u.y, u.x);
        double rhs_u = rhs.at<double>(u.y, u.x);

        // 调试：输出当前处理的节点（每100次迭代）
        if (iteration % 1000 == 0) {
            std::cout << "[D* Lite] 处理节点(" << u.x << "," << u.y << ") "
                << "Key=[" << u.k1 << "," << u.k2 << "] "
                << "g=" << g_old << " rhs=" << rhs_u << std::endl;
        }

        // 过一致性：g > rhs，降低g值
        if (g_old > rhs_u + EPS) {
            g.at<double>(u.y, u.x) = rhs_u;

            // 向所有前驱传播（所有可以到达u的节点）
            auto preds = getNbrs(u.x, u.y);
            for (auto& p : preds) {
                updateVertex(p.x, p.y);
            }
        }
        // 欠一致性：g < rhs，升高g值
        else if (g_old < rhs_u - EPS) {
            g.at<double>(u.y, u.x) = INF;

            // 重新更新自己和所有前驱
            updateVertex(u.x, u.y);

            auto preds = getNbrs(u.x, u.y);
            for (auto& p : preds) {
                updateVertex(p.x, p.y);
            }
        }
        // 否则 g == rhs，已经是consistent了，不需要处理

        // 检查终止条件
        double g_start = g.at<double>(start.y, start.x);
        double rhs_start = rhs.at<double>(start.y, start.x);

        if (std::abs(g_start - rhs_start) < EPS && g_start < INF &&
            open.top().k1 >= calcKey(start.x, start.y).first - EPS) {
            std::cout << "[D* Lite] 找到最短路径! 迭代次数: " << iteration << std::endl;
            std::cout << "[D* Lite] 起点g值: " << g_start << std::endl;
            break;
        }
    }

    // ========== 调试信息 #2: 主循环结束 ==========
    std::cout << "[D* Lite] ========== 主循环结束 ==========" << std::endl;
    std::cout << "[D* Lite] 总迭代次数: " << iteration << std::endl;
    std::cout << "[D* Lite] 队列剩余大小: " << open.size() << std::endl;

    double g_start_final = g.at<double>(start.y, start.x);
    double rhs_start_final = rhs.at<double>(start.y, start.x);
    std::cout << "[D* Lite] 起点最终状态: g=" << g_start_final << ", rhs=" << rhs_start_final << std::endl;

    // 检查是否有解
    if (g_start_final >= INF - 1) {
        std::cerr << "[D* Lite] ERROR: 无可行路径! 起点无法到达终点。" << std::endl;
        return {};
    }

    // ========== 路径提取 ==========
    std::cout << "[D* Lite] ========== 开始路径提取 ==========" << std::endl;

    std::vector<cv::Point> path;
    cv::Point cur = start;
    path.push_back(cur);

    int max_steps = rows * cols;
    int steps = 0;

    while (!(cur == goal) && steps < max_steps) {
        steps++;

        double best = INF;
        cv::Point next = cur;
        double best_cost = 0;

        // 在所有后继中选择 g(s') + c(s, s') 最小的
        auto succs = getNbrs(cur.x, cur.y);

        for (auto& s : succs) {
            double c = moveCost(cur.x, cur.y, s.x, s.y);
            if (c >= INF) continue;

            double g_s = g.at<double>(s.y, s.x);
            double val = c + g_s;

            if (val < best - EPS) {
                best = val;
                next = s;
                best_cost = c;
            }
        }

        // 调试：每10步输出一次
        if (steps % 10 == 0 || steps == 1) {
            std::cout << "[D* Lite] 路径提取步骤 " << steps << ": 从("
                << cur.x << "," << cur.y << ") -> ("
                << next.x << "," << next.y << ") 代价=" << best_cost
                << " 累计g=" << best << std::endl;
        }

        // 检查是否卡住
        if (next == cur) {
            std::cerr << "[D* Lite] ERROR: 路径提取卡住! 步骤 " << steps
                << " 无法找到下一个节点" << std::endl;
            std::cerr << "[D* Lite] 当前位置: (" << cur.x << "," << cur.y << ")" << std::endl;
            break;
        }

        cur = next;
        path.push_back(cur);
    }

    std::cout << "[D* Lite] 路径提取完成，共 " << steps << " 步，路径长度 " << path.size() << std::endl;

    if (!(cur == goal)) {
        std::cerr << "[D* Lite] WARNING: 路径未到达目标点!" << std::endl;
        std::cerr << "[D* Lite] 当前终点: (" << cur.x << "," << cur.y
            << ") 目标: (" << goal.x << "," << goal.y << ")" << std::endl;
    }

    return path;
}

/* =========================
 * Hybrid A*
 * 标准化版本：
 * 1) 连续状态 (x, y, yaw)
 * 2) 离散状态栅格 (ix, iy, yaw_bin) 做剪枝
 * 3) 单轨模型扩展 motion primitives
 * 4) 双启发式：
 *      - holonomic with obstacles : 2D Dijkstra
 *      - non-holonomic without obstacles : Reeds-Shepp distance
 * 5) 接近目标时使用 Reeds-Shepp one-shot 直接连接
 * 6) 输出为路径点序列 std::vector<cv::Point>
 *
 * 说明：
 * - 对外 start / goal 仍然是栅格坐标
 * - 所有长度参数先按“米”定义，再通过 map_resolution 转成栅格单位
 * - 当前固定地图分辨率 map_resolution = 0.1 m/cell
 * ========================= */

std::vector<cv::Point> PathPlanner::planHybridAStar(
    const cv::Mat& costmap,
    const cv::Point& start,
    const cv::Point& goal,
    double start_yaw,
    double goal_yaw
)
{
    // =========================
    // 0. 输入检查
    // =========================
    validateCostmapOrThrow(costmap, "planHybridAStar");
    validatePointOrThrow(costmap, start, "start", "planHybridAStar");
    validatePointOrThrow(costmap, goal, "goal", "planHybridAStar");

    if (isObstacle_(costmap, start.x, start.y))
        throw std::runtime_error("planHybridAStar: start is obstacle.");
    if (isObstacle_(costmap, goal.x, goal.y))
        throw std::runtime_error("planHybridAStar: goal is obstacle.");

    // =========================
    // 1. 内部固定参数
    //    所有长度参数先按“米”定义，再换算成“栅格”
    // =========================

    // 地图分辨率：0.1 m / cell
    const double map_resolution = 0.1;

    // ---------- 物理参数（米 / 度） ----------
    const double step_size_m = 0.2;            // 每个 motion primitive 的长度，单位：米
    const double wheel_base_m = 1.0;           // 车辆轴距，单位：米
    const double safe_radius_m = 0.4;          // 安全半径，单位：米
    const double goal_tolerance_m = 0.2;       // 终点距离阈值，单位：米
    const double motion_resolution_m = 0.1;    // primitive 内部碰撞检测采样分辨率，单位：米
    const double rs_discretization_m = 0.1;    // RS 路径离散采样分辨率，单位：米

    // ---------- 其他规划参数 ----------
    const int yaw_bins = 18;
    const double max_steer_deg = 30.0;
    const bool allow_reverse = true;
    const double cost_weight = 10.0;
    const double yaw_tolerance_deg = 5.0;

    // ---------- 单位换算：米 -> 栅格 ----------
    auto metersToCells = [&](double meters) -> double {
        return meters / map_resolution;
        };

    auto metersToCellsInt = [&](double meters) -> int {
        return std::max(1, static_cast<int>(std::round(meters / map_resolution)));
        };

    const double step_size = std::max(0.5, metersToCells(step_size_m));
    const double wheel_base = std::max(1.0, metersToCells(wheel_base_m));
    const int safe_radius = metersToCellsInt(safe_radius_m);
    const double goal_tolerance = metersToCells(goal_tolerance_m);
    const double motion_resolution = std::max(0.25, metersToCells(motion_resolution_m));
    const double rs_discretization = std::max(0.25, metersToCells(rs_discretization_m));

    if (!isSafe_(costmap, start.x, start.y, safe_radius))
        throw std::runtime_error("planHybridAStar: start is not safe.");
    if (!isSafe_(costmap, goal.x, goal.y, safe_radius))
        throw std::runtime_error("planHybridAStar: goal is not safe.");

    if (start == goal) {
        return { start };
    }

    // 若未给定起点/终点朝向，则自动补默认值
    if (std::isnan(start_yaw)) {
        start_yaw = std::atan2(goal.y - start.y, goal.x - start.x);
    }
    if (std::isnan(goal_yaw)) {
        goal_yaw = start_yaw;
    }

    start_yaw = normalizeAngle(start_yaw);
    goal_yaw = normalizeAngle(goal_yaw);

    const double max_steer = deg2rad(max_steer_deg);
    if (std::abs(std::tan(max_steer)) < 1e-9) {
        throw std::runtime_error("planHybridAStar: max_steer_deg is too small.");
    }

    const double yaw_tolerance = deg2rad(yaw_tolerance_deg);

    const int w = costmap.cols;
    const int h = costmap.rows;
    const int total_states = w * h * yaw_bins;

    // Reeds-Shepp 构造参数是曲率 kappa，不是半径
    const double max_curvature = std::tan(max_steer) / wheel_base;

    // =========================
    // 2. 工具函数
    // =========================

    // 把连续 yaw 离散为 [0, yaw_bins-1]
    auto yawToBin = [&](double yaw) -> int {
        double a = normalizeAngle(yaw);
        double t = (a + PI) / (2.0 * PI);  // [0,1)
        int bin = static_cast<int>(std::floor(t * yaw_bins));
        if (bin < 0) bin = 0;
        if (bin >= yaw_bins) bin = yaw_bins - 1;
        return bin;
        };

    // 3D 离散状态 -> 一维索引
    auto stateIndex = [&](int x, int y, int yaw_bin) -> int {
        return (y * w + x) * yaw_bins + yaw_bin;
        };

    // 2D 栅格 -> 一维索引
    auto gridIndex = [&](int x, int y) -> int {
        return y * w + x;
        };

    // 是否可通行：在地图内、不是障碍、且满足安全半径
    auto isTraversable = [&](int x, int y) -> bool {
        if (!inRange(costmap, x, y)) return false;
        if (isObstacle_(costmap, x, y)) return false;
        if (!isSafe_(costmap, x, y, safe_radius)) return false;
        return true;
        };

    // 读取代价图代价
    auto cellCost = [&](int x, int y) -> double {
        return costmap.at<double>(y, x);
        };

    // 向路径尾部追加点，自动去重
    auto appendPointUnique = [&](std::vector<cv::Point>& pts, const cv::Point& p) {
        if (pts.empty() || pts.back() != p) {
            pts.push_back(p);
        }
        };

    // =========================
    // 3. 节点定义
    // =========================
    struct Node {
        // 连续状态
        double x = 0.0;
        double y = 0.0;
        double yaw = 0.0;

        // A* 代价
        double g = INF_COST;
        double f = INF_COST;

        // 离散状态（用于剪枝）
        int ix = -1;
        int iy = -1;
        int ib = -1;

        // 父节点
        int parent = -1;

        // 到达当前节点时的控制信息
        double steer = 0.0;
        bool reverse = false;

        // 从父节点到当前节点这一段 primitive 的离散采样点
        std::vector<cv::Point> segment_points;
    };

    struct OpenNode {
        double f = INF_COST;
        int node_id = -1;
    };

    struct OpenCmp {
        bool operator()(const OpenNode& a, const OpenNode& b) const {
            return a.f > b.f;
        }
    };

    // =========================
    // 4. 预计算 holonomic-with-obstacles 启发式
    //    从 goal 反向做一次 2D Dijkstra
    // =========================
    std::vector<double> holo_heuristic(w * h, INF_COST);

    {
        struct DNode {
            double g = INF_COST;
            int x = -1;
            int y = -1;
        };

        struct DCmp {
            bool operator()(const DNode& a, const DNode& b) const {
                return a.g > b.g;
            }
        };

        std::priority_queue<DNode, std::vector<DNode>, DCmp> pq;

        holo_heuristic[gridIndex(goal.x, goal.y)] = 0.0;
        pq.push({ 0.0, goal.x, goal.y });

        static const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
        static const int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

        while (!pq.empty()) {
            DNode cur = pq.top();
            pq.pop();

            int cidx = gridIndex(cur.x, cur.y);
            if (cur.g > holo_heuristic[cidx] + EPS) continue;

            for (int k = 0; k < 8; ++k) {
                int nx = cur.x + dx[k];
                int ny = cur.y + dy[k];

                if (!inRange(costmap, nx, ny)) continue;
                if (!isTraversable(nx, ny)) continue;

                // 启发式只保留几何意义，不额外叠加 terrain cost
                double move_cost = (dx[k] == 0 || dy[k] == 0) ? 1.0 : std::sqrt(2.0);
                double ng = cur.g + move_cost;

                int nidx = gridIndex(nx, ny);
                if (ng + EPS < holo_heuristic[nidx]) {
                    holo_heuristic[nidx] = ng;
                    pq.push({ ng, nx, ny });
                }
            }
        }
    }

    // =========================
    // 5. Reeds-Shepp 对象
    //    注意构造参数是曲率 kappa
    // =========================
    steering::Reeds_Shepp_State_Space rs(max_curvature, rs_discretization);

    // non-holonomic-without-obstacles 启发式
    auto rsHeuristic = [&](double x, double y, double yaw) -> double {
        steering::State s0, s1;
        s0.x = x;
        s0.y = y;
        s0.theta = yaw;
        s0.kappa = 0.0;
        s0.d = 0.0;

        s1.x = goal.x;
        s1.y = goal.y;
        s1.theta = goal_yaw;
        s1.kappa = 0.0;
        s1.d = 0.0;

        return rs.get_distance(s0, s1);
        };

    // 双启发式：max(holo_with_obstacles, rs_without_obstacles)
    auto heuristic = [&](double x, double y, double yaw, int ix, int iy) -> double {
        double h_holo = INF_COST;
        if (inRange(costmap, ix, iy)) {
            h_holo = holo_heuristic[gridIndex(ix, iy)];
        }

        double h_rs = rsHeuristic(x, y, yaw);

        // 若 2D Dijkstra 查不到，退化为 RS 启发式
        if (h_holo >= INF_COST * 0.5) {
            return h_rs;
        }
        return std::max(h_holo, h_rs);
        };

    // =========================
    // 6. Reeds-Shepp analytical expansion
    //    尝试从当前节点 one-shot 连接到目标
    // =========================
    auto tryAnalyticalExpansion = [&](const Node& cur, std::vector<cv::Point>& rs_points_out) -> bool {
        rs_points_out.clear();

        // 不允许倒车时，不做 RS one-shot
        if (!allow_reverse) {
            return false;
        }

        steering::State s0, s1;
        s0.x = cur.x;
        s0.y = cur.y;
        s0.theta = cur.yaw;
        s0.kappa = 0.0;
        s0.d = 0.0;

        s1.x = goal.x;
        s1.y = goal.y;
        s1.theta = goal_yaw;
        s1.kappa = 0.0;
        s1.d = 0.0;

        std::vector<steering::State> rs_path = rs.get_path(s0, s1);
        if (rs_path.empty()) {
            return false;
        }

        for (const auto& st : rs_path) {
            int gx = static_cast<int>(std::round(st.x));
            int gy = static_cast<int>(std::round(st.y));

            if (!isTraversable(gx, gy)) {
                rs_points_out.clear();
                return false;
            }

            appendPointUnique(rs_points_out, cv::Point(gx, gy));
        }

        if (rs_points_out.empty()) {
            return false;
        }

        appendPointUnique(rs_points_out, goal);
        return true;
        };

    // =========================
    // 7. A* 数据结构
    // =========================
    std::priority_queue<OpenNode, std::vector<OpenNode>, OpenCmp> open;
    std::vector<double> best_g(total_states, INF_COST);
    std::vector<unsigned char> closed(total_states, 0);
    std::vector<Node> nodes;
    nodes.reserve(100000);

    auto addNode = [&](const Node& n) -> int {
        nodes.push_back(n);
        return static_cast<int>(nodes.size()) - 1;
        };

    // 起点节点
    Node start_node;
    start_node.x = static_cast<double>(start.x);
    start_node.y = static_cast<double>(start.y);
    start_node.yaw = start_yaw;
    start_node.g = 0.0;
    start_node.ix = start.x;
    start_node.iy = start.y;
    start_node.ib = yawToBin(start_yaw);
    start_node.f = heuristic(start_node.x, start_node.y, start_node.yaw, start_node.ix, start_node.iy);
    start_node.parent = -1;
    start_node.steer = 0.0;
    start_node.reverse = false;
    start_node.segment_points.push_back(start);

    int start_id = addNode(start_node);
    best_g[stateIndex(start_node.ix, start_node.iy, start_node.ib)] = 0.0;
    open.push(OpenNode{ start_node.f, start_id });

    // 控制输入离散化：前进/后退 + 多个转角
    std::vector<double> steer_set = {
        -max_steer,
        -max_steer / 2.0,
         0.0,
         max_steer / 2.0,
         max_steer
    };

    std::vector<int> dir_set = allow_reverse ? std::vector<int>{ 1, -1 } : std::vector<int>{ 1 };

    int goal_node_id = -1;

    // =========================
    // 8. Hybrid A* 主循环
    // =========================

    int expanded_count = 0;
    const int max_expanded_nodes = 100000;

    auto t0 = std::chrono::steady_clock::now();
    const double max_search_time_ms = 10000.0;

    while (!open.empty()) {

        auto now = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - t0).count();

        if (expanded_count >= max_expanded_nodes || elapsed_ms >= max_search_time_ms) {
            break;
        }
        OpenNode on = open.top();
        open.pop();

        if (on.node_id < 0 || on.node_id >= static_cast<int>(nodes.size())) continue;
        Node cur = nodes[on.node_id];

        int cur_flat = stateIndex(cur.ix, cur.iy, cur.ib);
        if (closed[cur_flat]) continue;
        closed[cur_flat] = 1;

        ++expanded_count;

        // 当前节点到目标的距离和航向误差
        double dist_to_goal = std::hypot(cur.x - goal.x, cur.y - goal.y);
        double yaw_err = std::abs(normalizeAngle(cur.yaw - goal_yaw));

        // -----------------------------------------
        // 8.1 接近终点时尝试 RS one-shot
        // -----------------------------------------
        {
            double rs_try_distance = std::max(6.0, 6.0 * step_size);

            if (dist_to_goal <= rs_try_distance) {
                std::vector<cv::Point> rs_tail_points;
                if (tryAnalyticalExpansion(cur, rs_tail_points)) {
                    Node goal_node;
                    goal_node.x = static_cast<double>(goal.x);
                    goal_node.y = static_cast<double>(goal.y);
                    goal_node.yaw = goal_yaw;
                    goal_node.g = cur.g + dist_to_goal;
                    goal_node.ix = goal.x;
                    goal_node.iy = goal.y;
                    goal_node.ib = yawToBin(goal_yaw);
                    goal_node.f = goal_node.g;
                    goal_node.parent = on.node_id;
                    goal_node.steer = cur.steer;
                    goal_node.reverse = cur.reverse;
                    goal_node.segment_points = rs_tail_points;

                    goal_node_id = addNode(goal_node);
                    break;
                }
            }
        }

        // -----------------------------------------
        // 8.2 普通终止条件
        // -----------------------------------------
        if (dist_to_goal <= goal_tolerance && yaw_err <= yaw_tolerance) {
            goal_node_id = on.node_id;
            break;
        }

        // -----------------------------------------
        // 8.3 扩展 motion primitives
        // -----------------------------------------
        for (int dir : dir_set) {
            for (double steer : steer_set) {
                // primitive 总位移（栅格单位）
                double ds = dir * step_size;

                // 单轨模型曲率 k = tan(delta) / L
                double curvature = std::tan(steer) / wheel_base;

                double nx = cur.x;
                double ny = cur.y;
                double nyaw = cur.yaw;

                // 将一个 primitive 再细分成若干小步，逐点碰撞检测
                int sub_steps = std::max(2, static_cast<int>(std::ceil(std::abs(step_size) / motion_resolution)));
                double sub_ds = ds / static_cast<double>(sub_steps);

                bool collision = false;
                double traversability_cost = 0.0;

                std::vector<cv::Point> primitive_points;

                for (int s = 0; s < sub_steps; ++s) {
                    // 单轨模型离散积分
                    nx += sub_ds * std::cos(nyaw);
                    ny += sub_ds * std::sin(nyaw);
                    nyaw = normalizeAngle(nyaw + sub_ds * curvature);

                    int gx = static_cast<int>(std::round(nx));
                    int gy = static_cast<int>(std::round(ny));

                    if (!inRange(costmap, gx, gy) || !isTraversable(gx, gy)) {
                        collision = true;
                        break;
                    }

                    traversability_cost += cellCost(gx, gy);
                    appendPointUnique(primitive_points, cv::Point(gx, gy));
                }

                if (collision) continue;
                if (primitive_points.empty()) continue;

                int ix = static_cast<int>(std::round(nx));
                int iy = static_cast<int>(std::round(ny));
                int ib = yawToBin(nyaw);

                if (!inRange(costmap, ix, iy)) continue;
                if (!isTraversable(ix, iy)) continue;

                int flat = stateIndex(ix, iy, ib);
                if (closed[flat]) continue;

                // -----------------------------------------
                // 8.4 代价函数 g(n)
                // -----------------------------------------
                // 包括：
                // 1) 路径长度代价
                // 2) 地形代价
                // 3) 转向代价
                // 4) 转向变化代价
                // 5) 倒车惩罚
                // 6) 换挡惩罚
                double motion_cost = std::abs(step_size);
                double terrain_cost = cost_weight * traversability_cost / static_cast<double>(sub_steps);
                double steer_cost = 0.2 * std::abs(steer);
                double steer_change_cost = 0.25 * std::abs(steer - cur.steer);
                double reverse_cost = (dir < 0) ? 2.0 : 0.0;
                double gear_switch_cost = ((dir < 0) != cur.reverse) ? 3.0 : 0.0;

                double new_g = cur.g
                    + motion_cost
                    + terrain_cost
                    + steer_cost
                    + steer_change_cost
                    + reverse_cost
                    + gear_switch_cost;

                if (new_g + EPS < best_g[flat]) {
                    best_g[flat] = new_g;

                    Node next;
                    next.x = nx;
                    next.y = ny;
                    next.yaw = nyaw;
                    next.g = new_g;
                    next.ix = ix;
                    next.iy = iy;
                    next.ib = ib;
                    next.f = new_g + heuristic(nx, ny, nyaw, ix, iy);
                    next.parent = on.node_id;
                    next.steer = steer;
                    next.reverse = (dir < 0);
                    next.segment_points = primitive_points;

                    int nid = addNode(next);
                    open.push(OpenNode{ next.f, nid });
                }
            }
        }
    }

    // 未找到可行路径
    if (goal_node_id < 0) {
        return {};
    }

    // =========================
    // 9. 回溯路径
    // =========================
    // 不只是回溯节点中心点，
    // 而是把每条 primitive 内部采样点一起恢复出来
    std::vector<int> node_chain;
    {
        int cur_id = goal_node_id;
        while (cur_id >= 0) {
            node_chain.push_back(cur_id);
            cur_id = nodes[cur_id].parent;
        }
        std::reverse(node_chain.begin(), node_chain.end());
    }

    std::vector<cv::Point> path;
    for (int nid : node_chain) {
        const Node& n = nodes[nid];
        for (const auto& p : n.segment_points) {
            appendPointUnique(path, p);
        }
    }

    // 保底：确保首点是 start
    if (path.empty() || path.front() != start) {
        path.insert(path.begin(), start);
    }

    // 保底：确保末点贴近 goal
    if (std::hypot(path.back().x - goal.x, path.back().y - goal.y) > goal_tolerance) {
        appendPointUnique(path, goal);
    }

    return path;
}

/* =========================
 * Bidirectional A* (双向A*)
 * ========================= */

std::vector<cv::Point> PathPlanner::planBidirectionalAStar(
    const cv::Mat& costmap,
    const cv::Point& start,
    const cv::Point& goal,
    int safe_radius,
    double cost_weight
)
{
    // ========== 参数检查 ==========
    if (costmap.empty() || costmap.type() != CV_64FC1)
        throw std::runtime_error("planBidirectionalAStar: costmap must be CV_64FC1.");

    const int w = costmap.cols;
    const int h = costmap.rows;

    if (start.x < 0 || start.y < 0 || start.x >= w || start.y >= h)
        throw std::runtime_error("planBidirectionalAStar: start out of range.");
    if (goal.x < 0 || goal.y < 0 || goal.x >= w || goal.y >= h)
        throw std::runtime_error("planBidirectionalAStar: goal out of range.");

    if (isObstacle_(costmap, start.x, start.y))
        throw std::runtime_error("planBidirectionalAStar: start is obstacle.");
    if (isObstacle_(costmap, goal.x, goal.y))
        throw std::runtime_error("planBidirectionalAStar: goal is obstacle.");

    // 起点终点相同
    if (start == goal) {
        return { start };
    }

    // 八方向
    const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    const int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const double step_cost[8] = {
        1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0),
        1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0)
    };

    // ========== 数据结构 ==========
    // 正向搜索（起点→终点）
    cv::Mat g_forward(h, w, CV_64FC1, cv::Scalar(INF_COST));
    cv::Mat parent_x_forward(h, w, CV_32S, cv::Scalar(-1));
    cv::Mat parent_y_forward(h, w, CV_32S, cv::Scalar(-1));

    // 反向搜索（终点→起点）
    cv::Mat g_backward(h, w, CV_64FC1, cv::Scalar(INF_COST));
    cv::Mat parent_x_backward(h, w, CV_32S, cv::Scalar(-1));
    cv::Mat parent_y_backward(h, w, CV_32S, cv::Scalar(-1));

    // 节点是否在对应搜索中被访问
    cv::Mat visited_forward(h, w, CV_8U, cv::Scalar(0));
    cv::Mat visited_backward(h, w, CV_8U, cv::Scalar(0));

    // 优先队列节点: (f, g, x, y)
    // 使用g作为第二关键字，保证f相同时g小的先出队
    using Node = std::tuple<double, double, int, int>;

    // 正向队列（小顶堆）
    std::priority_queue<Node, std::vector<Node>, std::greater<>> open_forward;
    // 反向队列
    std::priority_queue<Node, std::vector<Node>, std::greater<>> open_backward;

    // 启发函数
    auto heuristic = [&](int x1, int y1, int x2, int y2) -> double {
        return std::hypot(x1 - x2, y1 - y2);
        };

    // ========== 初始化 ==========
    // 正向：从起点开始
    g_forward.at<double>(start.y, start.x) = 0.0;
    double f_start = heuristic(start.x, start.y, goal.x, goal.y);
    open_forward.emplace(f_start, 0.0, start.x, start.y);

    // 反向：从终点开始
    g_backward.at<double>(goal.y, goal.x) = 0.0;
    double f_goal = heuristic(goal.x, goal.y, start.x, start.y);  // 对称
    open_backward.emplace(f_goal, 0.0, goal.x, goal.y);

    // 相遇点记录
    cv::Point meeting_point(-1, -1);
    double best_path_cost = INF_COST;

    // ========== 主循环 ==========
    int iteration = 0;
    const int MAX_ITERATIONS = w * h * 2;

    while ((!open_forward.empty() || !open_backward.empty()) && iteration < MAX_ITERATIONS) {
        iteration++;

        // 选择扩展哪一侧：优先扩展f值小的一侧（交替策略也可）
        bool expand_forward = true;

        if (open_forward.empty()) {
            expand_forward = false;
        }
        else if (!open_backward.empty()) {
            // 比较两个队列的堆顶f值，选小的
            auto [f_f, g_f, x_f, y_f] = open_forward.top();
            auto [f_b, g_b, x_b, y_b] = open_backward.top();
            expand_forward = (f_f <= f_b);
        }

        if (expand_forward) {
            // ========== 扩展正向搜索 ==========
            auto [f, g, cx, cy] = open_forward.top();
            open_forward.pop();

            // 已访问过，跳过
            if (visited_forward.at<uchar>(cy, cx)) continue;
            visited_forward.at<uchar>(cy, cx) = 1;

            // 检查是否与反向搜索相遇
            if (visited_backward.at<uchar>(cy, cx)) {
                // 找到相遇点，计算总代价
                double total_cost = g_forward.at<double>(cy, cx) +
                    g_backward.at<double>(cy, cx);
                if (total_cost < best_path_cost) {
                    best_path_cost = total_cost;
                    meeting_point = cv::Point(cx, cy);
                }
                // 继续搜索，可能找到更优的相遇点
                // 当两侧的最小f值之和 >= best_path_cost 时可以停止
            }

            // 扩展8邻域
            for (int dir = 0; dir < 8; ++dir) {
                int nx = cx + dx[dir];
                int ny = cy + dy[dir];

                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                if (!isSafe_(costmap, nx, ny, safe_radius)) continue;
                if (visited_forward.at<uchar>(ny, nx)) continue;

                double new_g = g_forward.at<double>(cy, cx) +
                    step_cost[dir] +
                    cost_weight * costmap.at<double>(ny, nx);

                if (new_g + EPS < g_forward.at<double>(ny, nx)) {
                    g_forward.at<double>(ny, nx) = new_g;
                    parent_x_forward.at<int>(ny, nx) = cx;
                    parent_y_forward.at<int>(ny, nx) = cy;

                    double new_f = new_g + heuristic(nx, ny, goal.x, goal.y);
                    open_forward.emplace(new_f, new_g, nx, ny);
                }
            }
        }
        else {
            // ========== 扩展反向搜索 ==========
            auto [f, g, cx, cy] = open_backward.top();
            open_backward.pop();

            if (visited_backward.at<uchar>(cy, cx)) continue;
            visited_backward.at<uchar>(cy, cx) = 1;

            // 检查是否与正向搜索相遇
            if (visited_forward.at<uchar>(cy, cx)) {
                double total_cost = g_forward.at<double>(cy, cx) +
                    g_backward.at<double>(cy, cx);
                if (total_cost < best_path_cost) {
                    best_path_cost = total_cost;
                    meeting_point = cv::Point(cx, cy);
                }
            }

            // 扩展8邻域（注意：反向搜索的"目标"是起点）
            for (int dir = 0; dir < 8; ++dir) {
                int nx = cx + dx[dir];
                int ny = cy + dy[dir];

                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                if (!isSafe_(costmap, nx, ny, safe_radius)) continue;
                if (visited_backward.at<uchar>(ny, nx)) continue;

                double new_g = g_backward.at<double>(cy, cx) +
                    step_cost[dir] +
                    cost_weight * costmap.at<double>(ny, nx);

                if (new_g + EPS < g_backward.at<double>(ny, nx)) {
                    g_backward.at<double>(ny, nx) = new_g;
                    parent_x_backward.at<int>(ny, nx) = cx;
                    parent_y_backward.at<int>(ny, nx) = cy;

                    // 反向搜索的启发函数指向起点
                    double new_f = new_g + heuristic(nx, ny, start.x, start.y);
                    open_backward.emplace(new_f, new_g, nx, ny);
                }
            }
        }

        // 终止条件：如果找到相遇点，且两侧的最小f值之和 >= 当前最优路径
        if (meeting_point.x >= 0 && !open_forward.empty() && !open_backward.empty()) {
            auto [f_f, g_f, x_f, y_f] = open_forward.top();
            auto [f_b, g_b, x_b, y_b] = open_backward.top();

            // 如果两侧的最小f值之和 >= best_path_cost，可以终止
            // 因为任何新路径的代价都不会更优
            if (f_f + f_b >= best_path_cost - EPS) {
                break;
            }
        }
    }

    // ========== 检查是否找到路径 ==========
    if (meeting_point.x < 0) {
        return {};  // 无路径
    }

    // ========== 路径重建 ==========
    std::vector<cv::Point> path;

    // 1. 从相遇点回溯到起点（使用正向父指针）
    cv::Point cur = meeting_point;
    while (!(cur == start)) {
        path.push_back(cur);
        int px = parent_x_forward.at<int>(cur.y, cur.x);
        int py = parent_y_forward.at<int>(cur.y, cur.x);

        if (px < 0 || py < 0) break;  // 安全检查
        cur = cv::Point(px, py);
    }
    path.push_back(start);
    std::reverse(path.begin(), path.end());

    // 2. 从相遇点回溯到终点（使用反向父指针）
    cur = meeting_point;
    while (!(cur == goal)) {
        int px = parent_x_backward.at<int>(cur.y, cur.x);
        int py = parent_y_backward.at<int>(cur.y, cur.x);

        if (px < 0 || py < 0) break;
        cur = cv::Point(px, py);

        if (cur == meeting_point) break;  // 防止循环
        path.push_back(cur);
    }

    return path;
}